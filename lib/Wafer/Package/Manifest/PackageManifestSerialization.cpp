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
      json.attribute("module_format", manifest.moduleFormat);
    });
    json.attributeObject("launch", [&] {
      json.attribute("kind", "kernel");
      const KernelRuntimeLaunchContract &kernel = manifest.launch.getKernel();
      json.attribute("form", stringifyKernelLaunchForm(kernel.form));
      json.attribute("entry_abi", stringifyKernelEntryABI(kernel.entryABI));
      json.attributeArray("phases", [&] {
        for (RuntimeLaunchPhaseRole phase : manifest.launch.getPhases())
          json.value(stringifyRuntimeLaunchPhaseRole(phase));
      });
    });
    json.attribute("card_count", manifest.cardCount);
    json.attribute("tile_count", manifest.tileCount);
    json.attributeObject("program_data", [&] {
      json.attribute("relative_path", manifest.programData.relativePath);
      json.attribute("total_bytes", int64_t(manifest.programData.totalBytes));
      json.attribute("base_alignment",
                     int64_t(manifest.programData.baseAlignment));
      json.attribute("digest", manifest.programData.digest);
    });
    json.attributeArray("program_tensors", [&] {
      for (const ProgramTensorRecord &record : manifest.programTensors)
        json.object([&] {
          json.attribute("id", int64_t(record.id.getValue()));
          json.attribute("role", stringifyProgramTensorRole(record.role));
          json.attribute("role_index", record.roleIndex);
          json.attribute("dtype", stringifyProgramElementType(record.dtype));
          json.attributeArray("global_shape", [&] {
            for (int64_t dimension : record.globalShape)
              json.value(dimension);
          });
          json.attributeArray("local_shape", [&] {
            for (int64_t dimension : record.localShape)
              json.value(dimension);
          });
          json.attributeArray("slice_offsets", [&] {
            for (int64_t dimension : record.sliceOffsets)
              json.value(dimension);
          });
          json.attributeArray("slice_sizes", [&] {
            for (int64_t dimension : record.sliceSizes)
              json.value(dimension);
          });
        });
    });
    json.attributeArray("target_tensors", [&] {
      for (const TargetTensorRecord &record : manifest.targetTensors)
        json.object([&] {
          json.attribute("id", int64_t(record.id.getValue()));
          json.attribute("program_tensor",
                         int64_t(record.programTensor.getValue()));
          json.attribute("dtype", stringifyLogicalFormat(record.dtype));
          json.attribute("layout", stringifyPackageMemLayout(record.layout));
          json.attributeArray("shape", [&] {
            for (int64_t dimension : record.shape)
              json.value(dimension);
          });
          json.attribute("bytes", int64_t(record.bytes));
          json.attribute("alignment", int64_t(record.alignment));
          json.attribute("file_offset", int64_t(record.fileOffset));
        });
    });
    auto writePorts = [&](llvm::StringRef table,
                          llvm::ArrayRef<ExternalPortRecord> ports) {
      json.attributeArray(table, [&] {
        for (const ExternalPortRecord &port : ports)
          json.object([&] {
            json.attribute("id", int64_t(port.id.getValue()));
            json.attribute("role_index", port.roleIndex);
            json.attribute("logical_dtype",
                           stringifyProgramElementType(port.logicalDtype));
            json.attributeArray("logical_shape", [&] {
              for (int64_t dimension : port.logicalShape)
                json.value(dimension);
            });
            json.attribute("dtype", stringifyLogicalFormat(port.dtype));
            json.attribute("layout", stringifyPackageMemLayout(port.layout));
            json.attributeArray("shape", [&] {
              for (int64_t dimension : port.shape)
                json.value(dimension);
            });
            json.attribute("bytes", int64_t(port.bytes));
            json.attribute("alignment", int64_t(port.alignment));
          });
      });
    };
    writePorts("inputs", manifest.inputs);
    writePorts("outputs", manifest.outputs);
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
          json.attribute("launch_slot", int64_t(entry.launchSlot.getValue()));
          json.attribute("module", int64_t(entry.module.getValue()));
          json.attributeArray("arguments", [&] {
            for (const TileEntryArgumentRecord &argument : entry.arguments)
              json.object([&] {
                json.attribute("ordinal", int64_t(argument.ordinal));
                if (const auto *reference = std::get_if<ExternalInputArgument>(
                        &argument.reference)) {
                  json.attribute("kind", "external_input");
                  json.attribute("port", int64_t(reference->port.getValue()));
                } else if (const auto *reference =
                               std::get_if<TargetTensorArgument>(
                                   &argument.reference)) {
                  json.attribute("kind", "target_tensor");
                  json.attribute("tensor",
                                 int64_t(reference->tensor.getValue()));
                } else if (const auto *reference =
                               std::get_if<ExternalOutputArgument>(
                                   &argument.reference)) {
                  json.attribute("kind", "external_output");
                  json.attribute("port", int64_t(reference->port.getValue()));
                } else if (const auto *reference =
                               std::get_if<WorkspaceArgument>(
                                   &argument.reference)) {
                  json.attribute("kind", "workspace");
                  json.attribute("bytes", int64_t(reference->bytes));
                  json.attribute("alignment", int64_t(reference->alignment));
                } else if (const auto *reference =
                               std::get_if<SharedWorkspaceArgument>(
                                   &argument.reference)) {
                  json.attribute("kind", "shared_workspace");
                  json.attribute("resource", int64_t(reference->resource));
                  json.attribute("bytes", int64_t(reference->bytes));
                  json.attribute("alignment", int64_t(reference->alignment));
                  json.attribute("zero_initialize", reference->zeroInitialize);
                } else if (const auto *reference =
                               std::get_if<ProfileRecordArgument>(
                                   &argument.reference)) {
                  json.attribute("kind", "profile_record");
                  json.attribute("record_abi", reference->recordABI);
                  json.attribute("bytes", int64_t(reference->bytes));
                  json.attribute("alignment", int64_t(reference->alignment));
                } else {
                  const auto &statusReference =
                      std::get<TransportStatusArgument>(argument.reference);
                  json.attribute("kind", "transport_status");
                  json.attribute("status_abi", statusReference.statusABI);
                  json.attribute("bytes", int64_t(statusReference.bytes));
                  json.attribute("alignment",
                                 int64_t(statusReference.alignment));
                }
                json.attribute("access",
                               stringifyPackageAccessMode(argument.access));
              });
          });
          json.attribute("completion",
                         stringifyPackageEntryCompletionKind(entry.completion));
          json.attributeObject("transport", [&] {
            if (std::holds_alternative<NoTransportRequirements>(
                    entry.transport)) {
              json.attribute("kind", "none");
              return;
            }
            const auto &requirements =
                std::get<DirectDTETransportRequirements>(entry.transport);
            json.attribute("kind", "direct_dte");
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
