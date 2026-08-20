//===- PackageManifestJson.cpp - Typed manifest JSON decoding ------------===//

#include "PackageManifestInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace wafer::runtime {
namespace {

using detail::invalid;

bool exceedsJSONNesting(llvm::StringRef input, uint64_t maximum) {
  uint64_t depth = 0;
  bool inString = false;
  bool escaped = false;
  for (char character : input) {
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
    } else if (character == '{' || character == '[') {
      if (++depth > maximum)
        return true;
    } else if ((character == '}' || character == ']') && depth != 0) {
      --depth;
    }
  }
  return false;
}

llvm::Error requireExactFields(const llvm::json::Object &object,
                               std::initializer_list<llvm::StringRef> fields,
                               llvm::StringRef context) {
  llvm::StringSet<> expected;
  for (llvm::StringRef field : fields)
    expected.insert(field);
  for (const auto &member : object)
    if (!expected.contains(member.first))
      return invalid(context + " contains unknown field '" +
                     member.first.str() + "'");
  for (llvm::StringRef field : fields)
    if (object.find(field) == object.end())
      return invalid(context + " is missing field '" + field + "'");
  return llvm::Error::success();
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef field,
              llvm::StringRef context) {
  if (const llvm::json::Object *value = object.getObject(field))
    return value;
  return invalid(context + "." + field + " must be an object");
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef field,
             llvm::StringRef context) {
  if (const llvm::json::Array *value = object.getArray(field))
    return value;
  return invalid(context + "." + field + " must be an array");
}

llvm::Expected<std::string> requireString(const llvm::json::Object &object,
                                          llvm::StringRef field,
                                          llvm::StringRef context,
                                          const PackageParseLimits &limits,
                                          bool allowEmpty = false) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return invalid(context + "." + field + " must be a string");
  if ((!allowEmpty && value->empty()) || value->size() > limits.maxStringBytes)
    return invalid(context + "." + field + " has invalid length");
  return value->str();
}

llvm::Expected<int64_t> requireInteger(const llvm::json::Object &object,
                                       llvm::StringRef field,
                                       llvm::StringRef context) {
  std::optional<int64_t> value = object.getInteger(field);
  if (!value)
    return invalid(context + "." + field + " must be an integer");
  return *value;
}

llvm::Expected<uint64_t> requireUnsigned(const llvm::json::Object &object,
                                         llvm::StringRef field,
                                         llvm::StringRef context) {
  llvm::Expected<int64_t> value = requireInteger(object, field, context);
  if (!value)
    return value.takeError();
  if (*value < 0)
    return invalid(context + "." + field + " must be non-negative");
  return static_cast<uint64_t>(*value);
}

llvm::Expected<bool> requireBoolean(const llvm::json::Object &object,
                                    llvm::StringRef field,
                                    llvm::StringRef context) {
  std::optional<bool> value = object.getBoolean(field);
  if (!value)
    return invalid(context + "." + field + " must be a boolean");
  return *value;
}

llvm::Expected<ProgramTensorRole> parseProgramTensorRole(llvm::StringRef role) {
  if (role == "parameter")
    return ProgramTensorRole::Parameter;
  if (role == "constant")
    return ProgramTensorRole::Constant;
  return invalid("unsupported package program tensor role '" + role + "'");
}

llvm::Expected<PackageAccessMode> parseAccess(llvm::StringRef access) {
  if (access == "read_only")
    return PackageAccessMode::ReadOnly;
  if (access == "write_only")
    return PackageAccessMode::WriteOnly;
  if (access == "read_write")
    return PackageAccessMode::ReadWrite;
  return invalid("unsupported package access mode '" + access + "'");
}

llvm::Expected<PackageMemLayout> parseMemLayout(llvm::StringRef layout) {
  if (layout == "tensor")
    return PackageMemLayout::Tensor;
  if (layout == "ntensor")
    return PackageMemLayout::NTensor;
  if (layout == "cx")
    return PackageMemLayout::Cx;
  if (layout == "ncx")
    return PackageMemLayout::NCx;
  return invalid("unsupported package memory layout '" + layout + "'");
}

llvm::Expected<PackageModuleExportRole>
parseModuleExportRole(llvm::StringRef role) {
  if (role == "prepare")
    return PackageModuleExportRole::Prepare;
  if (role == "main")
    return PackageModuleExportRole::Main;
  return invalid("unsupported package module export role '" + role + "'");
}

llvm::Expected<PackageEntryCompletionKind>
parseEntryCompletionKind(llvm::StringRef kind) {
  if (kind == "return_after_local_drain")
    return PackageEntryCompletionKind::ReturnAfterLocalDrain;
  return invalid("unsupported package entry completion kind '" + kind + "'");
}

llvm::Expected<std::vector<RuntimeLaunchPhaseRole>>
parseRuntimeLaunchPhases(const llvm::json::Object &object,
                         llvm::StringRef context,
                         const PackageParseLimits &limits) {
  llvm::Expected<const llvm::json::Array *> phases =
      requireArray(object, "phases", context);
  if (!phases)
    return phases.takeError();
  if ((*phases)->empty() || (*phases)->size() > 2)
    return invalid(context + ".phases must contain one or two ordered roles");

  std::vector<RuntimeLaunchPhaseRole> result;
  result.reserve((*phases)->size());
  for (auto [index, value] : llvm::enumerate(**phases)) {
    std::optional<llvm::StringRef> spelling = value.getAsString();
    const std::string phaseContext =
        (context + ".phases[" + llvm::Twine(index) + "]").str();
    if (!spelling)
      return invalid(phaseContext + " must be a string");
    if (spelling->empty() || spelling->size() > limits.maxStringBytes)
      return invalid(phaseContext + " has invalid length");
    llvm::Expected<RuntimeLaunchPhaseRole> phase =
        parseRuntimeLaunchPhaseRole(*spelling);
    if (!phase)
      return phase.takeError();
    result.push_back(*phase);
  }
  return result;
}

llvm::Expected<RuntimeLaunchContract>
parseRuntimeLaunchContract(const llvm::json::Object &object,
                           llvm::StringRef context,
                           const PackageParseLimits &limits) {
  if (llvm::Error error = requireExactFields(
          object, {"kind", "form", "entry_abi", "phases"}, context))
    return std::move(error);
  llvm::Expected<std::string> kindSpelling =
      requireString(object, "kind", context, limits);
  if (!kindSpelling)
    return kindSpelling.takeError();
  if (*kindSpelling != "kernel")
    return invalid(context + ".kind must be 'kernel'");
  llvm::Expected<std::string> formSpelling =
      requireString(object, "form", context, limits);
  if (!formSpelling)
    return formSpelling.takeError();
  llvm::Expected<KernelLaunchForm> form = parseKernelLaunchForm(*formSpelling);
  if (!form)
    return form.takeError();
  llvm::Expected<std::string> entryABISpelling =
      requireString(object, "entry_abi", context, limits);
  if (!entryABISpelling)
    return entryABISpelling.takeError();
  llvm::Expected<KernelEntryABI> entryABI =
      parseKernelEntryABI(*entryABISpelling);
  if (!entryABI)
    return entryABI.takeError();
  llvm::Expected<std::vector<RuntimeLaunchPhaseRole>> phases =
      parseRuntimeLaunchPhases(object, context, limits);
  if (!phases)
    return phases.takeError();
  return RuntimeLaunchContract::createKernel(*form, *entryABI, *phases);
}

llvm::Expected<std::vector<int64_t>>
parseShape(const llvm::json::Object &object, llvm::StringRef field,
           llvm::StringRef context, const PackageParseLimits &limits) {
  llvm::Expected<const llvm::json::Array *> shape =
      requireArray(object, field, context);
  if (!shape)
    return shape.takeError();
  if ((*shape)->size() > limits.maxShapeRank)
    return invalid(context + "." + field + " exceeds rank limit");
  std::vector<int64_t> result;
  result.reserve((*shape)->size());
  for (const llvm::json::Value &dimensionValue : **shape) {
    std::optional<int64_t> dimension = dimensionValue.getAsInteger();
    if (!dimension || *dimension < 0)
      return invalid(context + "." + field +
                     " must contain non-negative integers");
    result.push_back(*dimension);
  }
  return result;
}

llvm::Expected<ProgramTensorRecord>
parseProgramTensorRecord(const llvm::json::Value &value, uint64_t index,
                         const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "program_tensors[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = requireExactFields(
          *object,
          {"id", "role", "role_index", "dtype", "global_shape", "local_shape",
           "slice_offsets", "slice_sizes"},
          context))
    return std::move(error);

  ProgramTensorRecord record;
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  record.id = ProgramTensorId(*id);
  llvm::Expected<std::string> roleText =
      requireString(*object, "role", context, limits);
  if (!roleText)
    return roleText.takeError();
  llvm::Expected<ProgramTensorRole> role = parseProgramTensorRole(*roleText);
  if (!role)
    return role.takeError();
  record.role = *role;
  llvm::Expected<int64_t> roleIndex =
      requireInteger(*object, "role_index", context);
  if (!roleIndex)
    return roleIndex.takeError();
  record.roleIndex = *roleIndex;
  llvm::Expected<std::string> dtype =
      requireString(*object, "dtype", context, limits);
  if (!dtype)
    return dtype.takeError();
  llvm::Expected<ProgramElementType> elementType =
      parseProgramElementType(*dtype);
  if (!elementType)
    return elementType.takeError();
  record.dtype = *elementType;
  llvm::Expected<std::vector<int64_t>> globalShape =
      parseShape(*object, "global_shape", context, limits);
  if (!globalShape)
    return globalShape.takeError();
  record.globalShape = std::move(*globalShape);
  llvm::Expected<std::vector<int64_t>> localShape =
      parseShape(*object, "local_shape", context, limits);
  if (!localShape)
    return localShape.takeError();
  record.localShape = std::move(*localShape);
  llvm::Expected<std::vector<int64_t>> sliceOffsets =
      parseShape(*object, "slice_offsets", context, limits);
  if (!sliceOffsets)
    return sliceOffsets.takeError();
  record.sliceOffsets = std::move(*sliceOffsets);
  llvm::Expected<std::vector<int64_t>> sliceSizes =
      parseShape(*object, "slice_sizes", context, limits);
  if (!sliceSizes)
    return sliceSizes.takeError();
  record.sliceSizes = std::move(*sliceSizes);
  return record;
}

llvm::Expected<TargetTensorRecord>
parseTargetTensorRecord(const llvm::json::Value &value, uint64_t index,
                        const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "target_tensors[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error =
          requireExactFields(*object,
                             {"id", "program_tensor", "dtype", "layout",
                              "shape", "bytes", "alignment", "file_offset"},
                             context))
    return std::move(error);

  TargetTensorRecord record;
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  record.id = TargetTensorId(*id);
  llvm::Expected<uint64_t> programTensor =
      requireUnsigned(*object, "program_tensor", context);
  if (!programTensor)
    return programTensor.takeError();
  record.programTensor = ProgramTensorId(*programTensor);
  llvm::Expected<std::string> dtype =
      requireString(*object, "dtype", context, limits);
  if (!dtype)
    return dtype.takeError();
  llvm::Expected<LogicalFormat> targetFormat = parseLogicalFormat(*dtype);
  if (!targetFormat)
    return targetFormat.takeError();
  record.dtype = *targetFormat;
  llvm::Expected<std::string> layoutText =
      requireString(*object, "layout", context, limits);
  if (!layoutText)
    return layoutText.takeError();
  llvm::Expected<PackageMemLayout> layout = parseMemLayout(*layoutText);
  if (!layout)
    return layout.takeError();
  record.layout = *layout;
  llvm::Expected<std::vector<int64_t>> shape =
      parseShape(*object, "shape", context, limits);
  if (!shape)
    return shape.takeError();
  record.shape = std::move(*shape);
  llvm::Expected<uint64_t> bytes = requireUnsigned(*object, "bytes", context);
  if (!bytes)
    return bytes.takeError();
  record.bytes = *bytes;
  llvm::Expected<uint64_t> alignment =
      requireUnsigned(*object, "alignment", context);
  if (!alignment)
    return alignment.takeError();
  record.alignment = *alignment;
  llvm::Expected<uint64_t> fileOffset =
      requireUnsigned(*object, "file_offset", context);
  if (!fileOffset)
    return fileOffset.takeError();
  record.fileOffset = *fileOffset;
  return record;
}

llvm::Expected<ProgramDataRecord>
parseProgramDataRecord(const llvm::json::Object &object,
                       const PackageParseLimits &limits) {
  if (llvm::Error error = requireExactFields(
          object, {"relative_path", "total_bytes", "base_alignment", "digest"},
          "manifest.program_data"))
    return std::move(error);
  ProgramDataRecord record;
  llvm::Expected<std::string> relativePath =
      requireString(object, "relative_path", "manifest.program_data", limits);
  if (!relativePath)
    return relativePath.takeError();
  record.relativePath = std::move(*relativePath);
  llvm::Expected<uint64_t> totalBytes =
      requireUnsigned(object, "total_bytes", "manifest.program_data");
  if (!totalBytes)
    return totalBytes.takeError();
  record.totalBytes = *totalBytes;
  llvm::Expected<uint64_t> baseAlignment =
      requireUnsigned(object, "base_alignment", "manifest.program_data");
  if (!baseAlignment)
    return baseAlignment.takeError();
  record.baseAlignment = *baseAlignment;
  llvm::Expected<std::string> digest =
      requireString(object, "digest", "manifest.program_data", limits);
  if (!digest)
    return digest.takeError();
  record.digest = std::move(*digest);
  return record;
}

llvm::Expected<ExternalPortRecord>
parseExternalPortRecord(const llvm::json::Value &value, uint64_t index,
                        llvm::StringRef tableName,
                        const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = tableName.str() + "[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = requireExactFields(
          *object,
          {"id", "role_index", "logical_dtype", "logical_shape", "dtype",
           "layout", "shape", "bytes", "alignment"},
          context))
    return std::move(error);

  ExternalPortRecord record;
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  record.id = PortId(*id);
  llvm::Expected<int64_t> roleIndex =
      requireInteger(*object, "role_index", context);
  if (!roleIndex)
    return roleIndex.takeError();
  record.roleIndex = *roleIndex;
  llvm::Expected<std::string> logicalDtype =
      requireString(*object, "logical_dtype", context, limits);
  if (!logicalDtype)
    return logicalDtype.takeError();
  llvm::Expected<ProgramElementType> logicalElementType =
      parseProgramElementType(*logicalDtype);
  if (!logicalElementType)
    return logicalElementType.takeError();
  record.logicalDtype = *logicalElementType;
  llvm::Expected<std::vector<int64_t>> logicalShape =
      parseShape(*object, "logical_shape", context, limits);
  if (!logicalShape)
    return logicalShape.takeError();
  record.logicalShape = std::move(*logicalShape);
  llvm::Expected<std::string> dtype =
      requireString(*object, "dtype", context, limits);
  if (!dtype)
    return dtype.takeError();
  llvm::Expected<LogicalFormat> targetFormat = parseLogicalFormat(*dtype);
  if (!targetFormat)
    return targetFormat.takeError();
  record.dtype = *targetFormat;
  llvm::Expected<std::string> layoutText =
      requireString(*object, "layout", context, limits);
  if (!layoutText)
    return layoutText.takeError();
  llvm::Expected<PackageMemLayout> layout = parseMemLayout(*layoutText);
  if (!layout)
    return layout.takeError();
  record.layout = *layout;
  llvm::Expected<std::vector<int64_t>> shape =
      parseShape(*object, "shape", context, limits);
  if (!shape)
    return shape.takeError();
  record.shape = std::move(*shape);
  llvm::Expected<uint64_t> bytes = requireUnsigned(*object, "bytes", context);
  if (!bytes)
    return bytes.takeError();
  record.bytes = *bytes;
  llvm::Expected<uint64_t> alignment =
      requireUnsigned(*object, "alignment", context);
  if (!alignment)
    return alignment.takeError();
  record.alignment = *alignment;
  return record;
}

llvm::Expected<PackageModuleRecord>
parseModuleRecord(const llvm::json::Value &value, uint64_t index,
                  const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "modules[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error = requireExactFields(
          *object, {"id", "path", "digest", "format", "exports"}, context))
    return std::move(error);
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  llvm::Expected<std::string> path =
      requireString(*object, "path", context, limits);
  if (!path)
    return path.takeError();
  llvm::Expected<std::string> digest =
      requireString(*object, "digest", context, limits);
  if (!digest)
    return digest.takeError();
  llvm::Expected<std::string> format =
      requireString(*object, "format", context, limits);
  if (!format)
    return format.takeError();
  llvm::Expected<const llvm::json::Array *> exports =
      requireArray(*object, "exports", context);
  if (!exports)
    return exports.takeError();
  if ((*exports)->size() > limits.maxRecords)
    return invalid(context + ".exports exceeds record limit");

  PackageModuleRecord record{ModuleId(*id),
                             std::move(*path),
                             std::move(*digest),
                             std::move(*format),
                             {}};
  for (auto [exportIndex, exportValue] : llvm::enumerate(**exports)) {
    const llvm::json::Object *moduleExport = exportValue.getAsObject();
    std::string exportContext =
        context + ".exports[" + std::to_string(exportIndex) + "]";
    if (!moduleExport)
      return invalid(exportContext + " must be an object");
    if (llvm::Error error = requireExactFields(
            *moduleExport, {"role", "symbol"}, exportContext))
      return std::move(error);
    llvm::Expected<std::string> roleText =
        requireString(*moduleExport, "role", exportContext, limits);
    if (!roleText)
      return roleText.takeError();
    llvm::Expected<PackageModuleExportRole> role =
        parseModuleExportRole(*roleText);
    if (!role)
      return role.takeError();
    llvm::Expected<std::string> symbol =
        requireString(*moduleExport, "symbol", exportContext, limits);
    if (!symbol)
      return symbol.takeError();
    record.exports.push_back({*role, std::move(*symbol)});
  }
  return record;
}

llvm::Expected<TransportRequirements>
parseTransportRequirements(const llvm::json::Object &object,
                           llvm::StringRef context,
                           const PackageParseLimits &limits) {
  llvm::Expected<std::string> kind =
      requireString(object, "kind", context, limits);
  if (!kind)
    return kind.takeError();
  if (*kind == "none") {
    if (llvm::Error error = requireExactFields(object, {"kind"}, context))
      return std::move(error);
    return TransportRequirements{NoTransportRequirements{}};
  }
  if (*kind != "direct_dte")
    return invalid(context + " has unsupported transport kind '" + *kind + "'");
  if (llvm::Error error = requireExactFields(
          object, {"kind", "status_abi", "host_watchdog_required"}, context))
    return std::move(error);
  llvm::Expected<std::string> statusABI =
      requireString(object, "status_abi", context, limits);
  if (!statusABI)
    return statusABI.takeError();
  llvm::Expected<bool> watchdog =
      requireBoolean(object, "host_watchdog_required", context);
  if (!watchdog)
    return watchdog.takeError();
  return TransportRequirements{
      DirectDTETransportRequirements{std::move(*statusABI), *watchdog}};
}

llvm::Expected<TileEntryArgumentReference>
parseTileEntryArgumentReference(const llvm::json::Object &object,
                                llvm::StringRef context,
                                const PackageParseLimits &limits) {
  llvm::Expected<std::string> kind =
      requireString(object, "kind", context, limits);
  if (!kind)
    return kind.takeError();
  if (*kind == "external_input") {
    if (llvm::Error error = requireExactFields(
            object, {"kind", "ordinal", "port", "access"}, context))
      return std::move(error);
    llvm::Expected<uint64_t> port = requireUnsigned(object, "port", context);
    if (!port)
      return port.takeError();
    return TileEntryArgumentReference{ExternalInputArgument{PortId(*port)}};
  }
  if (*kind == "target_tensor") {
    if (llvm::Error error = requireExactFields(
            object, {"kind", "ordinal", "tensor", "access"}, context))
      return std::move(error);
    llvm::Expected<uint64_t> tensor =
        requireUnsigned(object, "tensor", context);
    if (!tensor)
      return tensor.takeError();
    return TileEntryArgumentReference{
        TargetTensorArgument{TargetTensorId(*tensor)}};
  }
  if (*kind == "external_output") {
    if (llvm::Error error = requireExactFields(
            object, {"kind", "ordinal", "port", "access"}, context))
      return std::move(error);
    llvm::Expected<uint64_t> port = requireUnsigned(object, "port", context);
    if (!port)
      return port.takeError();
    return TileEntryArgumentReference{ExternalOutputArgument{PortId(*port)}};
  }
  if (*kind == "workspace") {
    if (llvm::Error error = requireExactFields(
            object, {"kind", "ordinal", "bytes", "alignment", "access"},
            context))
      return std::move(error);
    llvm::Expected<uint64_t> bytes = requireUnsigned(object, "bytes", context);
    if (!bytes)
      return bytes.takeError();
    llvm::Expected<uint64_t> alignment =
        requireUnsigned(object, "alignment", context);
    if (!alignment)
      return alignment.takeError();
    return TileEntryArgumentReference{WorkspaceArgument{*bytes, *alignment}};
  }
  if (*kind == "profile_record") {
    if (llvm::Error error = requireExactFields(
            object,
            {"kind", "ordinal", "record_abi", "bytes", "alignment", "access"},
            context))
      return std::move(error);
    llvm::Expected<std::string> recordABI =
        requireString(object, "record_abi", context, limits);
    if (!recordABI)
      return recordABI.takeError();
    llvm::Expected<uint64_t> bytes = requireUnsigned(object, "bytes", context);
    if (!bytes)
      return bytes.takeError();
    llvm::Expected<uint64_t> alignment =
        requireUnsigned(object, "alignment", context);
    if (!alignment)
      return alignment.takeError();
    return TileEntryArgumentReference{
        ProfileRecordArgument{std::move(*recordABI), *bytes, *alignment}};
  }
  if (*kind == "transport_status") {
    if (llvm::Error error = requireExactFields(
            object,
            {"kind", "ordinal", "status_abi", "bytes", "alignment", "access"},
            context))
      return std::move(error);
    llvm::Expected<std::string> statusABI =
        requireString(object, "status_abi", context, limits);
    if (!statusABI)
      return statusABI.takeError();
    llvm::Expected<uint64_t> bytes = requireUnsigned(object, "bytes", context);
    if (!bytes)
      return bytes.takeError();
    llvm::Expected<uint64_t> alignment =
        requireUnsigned(object, "alignment", context);
    if (!alignment)
      return alignment.takeError();
    return TileEntryArgumentReference{
        TransportStatusArgument{std::move(*statusABI), *bytes, *alignment}};
  }
  return invalid(context + " has unsupported argument kind '" + *kind + "'");
}

llvm::Expected<TileEntryArgumentRecord>
parseTileEntryArgument(const llvm::json::Value &value, uint64_t index,
                       const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "entries.arguments[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  TileEntryArgumentRecord record;
  llvm::Expected<uint64_t> ordinal =
      requireUnsigned(*object, "ordinal", context);
  if (!ordinal)
    return ordinal.takeError();
  record.ordinal = *ordinal;
  llvm::Expected<TileEntryArgumentReference> reference =
      parseTileEntryArgumentReference(*object, context, limits);
  if (!reference)
    return reference.takeError();
  record.reference = std::move(*reference);
  llvm::Expected<std::string> accessText =
      requireString(*object, "access", context, limits);
  if (!accessText)
    return accessText.takeError();
  llvm::Expected<PackageAccessMode> access = parseAccess(*accessText);
  if (!access)
    return access.takeError();
  record.access = *access;
  return record;
}

llvm::Expected<PackageEntrypointRecord>
parseEntrypointRecord(const llvm::json::Value &value, uint64_t index,
                      const PackageParseLimits &limits) {
  const llvm::json::Object *object = value.getAsObject();
  std::string context = "entries[" + std::to_string(index) + "]";
  if (!object)
    return invalid(context + " must be an object");
  if (llvm::Error error =
          requireExactFields(*object,
                             {"id", "card_id", "tile_id", "launch_slot",
                              "module", "arguments", "completion", "transport"},
                             context))
    return std::move(error);
  llvm::Expected<uint64_t> id = requireUnsigned(*object, "id", context);
  if (!id)
    return id.takeError();
  llvm::Expected<int64_t> cardId = requireInteger(*object, "card_id", context);
  if (!cardId)
    return cardId.takeError();
  llvm::Expected<int64_t> tileId = requireInteger(*object, "tile_id", context);
  if (!tileId)
    return tileId.takeError();
  llvm::Expected<uint64_t> launchSlot =
      requireUnsigned(*object, "launch_slot", context);
  if (!launchSlot)
    return launchSlot.takeError();
  llvm::Expected<uint64_t> module = requireUnsigned(*object, "module", context);
  if (!module)
    return module.takeError();
  llvm::Expected<const llvm::json::Array *> arguments =
      requireArray(*object, "arguments", context);
  if (!arguments)
    return arguments.takeError();
  llvm::Expected<std::string> completion =
      requireString(*object, "completion", context, limits);
  if (!completion)
    return completion.takeError();
  llvm::Expected<PackageEntryCompletionKind> completionKind =
      parseEntryCompletionKind(*completion);
  if (!completionKind)
    return completionKind.takeError();
  llvm::Expected<const llvm::json::Object *> transportObject =
      requireObject(*object, "transport", context);
  if (!transportObject)
    return transportObject.takeError();
  llvm::Expected<TransportRequirements> transport = parseTransportRequirements(
      **transportObject, context + ".transport", limits);
  if (!transport)
    return transport.takeError();
  if ((*arguments)->size() > limits.maxRecords)
    return invalid(context + ".arguments exceeds record limit");

  PackageEntrypointRecord record;
  record.id = EntryId(*id);
  record.cardId = CardId(*cardId);
  record.tileId = TileId(*tileId);
  record.launchSlot = LaunchSlotId(*launchSlot);
  record.module = ModuleId(*module);
  record.completion = *completionKind;
  record.transport = std::move(*transport);
  for (auto [argumentIndex, argumentValue] : llvm::enumerate(**arguments)) {
    llvm::Expected<TileEntryArgumentRecord> argument =
        parseTileEntryArgument(argumentValue, argumentIndex, limits);
    if (!argument)
      return argument.takeError();
    record.arguments.push_back(std::move(*argument));
  }
  return record;
}

} // namespace

llvm::Expected<PackageManifest>
detail::parseManifest(llvm::StringRef json, const PackageParseLimits &limits) {
  if (json.size() > limits.maxJSONBytes)
    return invalid("package manifest exceeds JSON byte limit");
  if (exceedsJSONNesting(json, limits.maxJSONNesting))
    return invalid("package manifest exceeds JSON nesting limit");
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json);
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("package manifest must be a JSON object");
  if (llvm::Error error = requireExactFields(
          *root,
          {"program", "target", "launch", "card_count", "tile_count",
           "program_data", "program_tensors", "target_tensors", "inputs",
           "outputs", "modules", "entries"},
          "manifest"))
    return std::move(error);
  llvm::Expected<const llvm::json::Object *> program =
      requireObject(*root, "program", "manifest");
  if (!program)
    return program.takeError();
  llvm::Expected<const llvm::json::Object *> target =
      requireObject(*root, "target", "manifest");
  if (!target)
    return target.takeError();
  llvm::Expected<const llvm::json::Object *> launch =
      requireObject(*root, "launch", "manifest");
  if (!launch)
    return launch.takeError();
  llvm::Expected<int64_t> cardCount =
      requireInteger(*root, "card_count", "manifest");
  if (!cardCount)
    return cardCount.takeError();
  llvm::Expected<int64_t> tileCount =
      requireInteger(*root, "tile_count", "manifest");
  if (!tileCount)
    return tileCount.takeError();
  llvm::Expected<const llvm::json::Object *> programData =
      requireObject(*root, "program_data", "manifest");
  if (!programData)
    return programData.takeError();
  llvm::Expected<const llvm::json::Array *> programTensors =
      requireArray(*root, "program_tensors", "manifest");
  if (!programTensors)
    return programTensors.takeError();
  llvm::Expected<const llvm::json::Array *> targetTensors =
      requireArray(*root, "target_tensors", "manifest");
  if (!targetTensors)
    return targetTensors.takeError();
  llvm::Expected<const llvm::json::Array *> inputs =
      requireArray(*root, "inputs", "manifest");
  if (!inputs)
    return inputs.takeError();
  llvm::Expected<const llvm::json::Array *> outputs =
      requireArray(*root, "outputs", "manifest");
  if (!outputs)
    return outputs.takeError();
  llvm::Expected<const llvm::json::Array *> modules =
      requireArray(*root, "modules", "manifest");
  if (!modules)
    return modules.takeError();
  llvm::Expected<const llvm::json::Array *> entries =
      requireArray(*root, "entries", "manifest");
  if (!entries)
    return entries.takeError();

  if (llvm::Error error =
          requireExactFields(**program, {"id"}, "manifest.program"))
    return std::move(error);
  if (llvm::Error error = requireExactFields(
          **target, {"identity", "runtime_abi", "module_format"},
          "manifest.target"))
    return std::move(error);
  llvm::Expected<uint64_t> programId =
      requireUnsigned(**program, "id", "manifest.program");
  if (!programId)
    return programId.takeError();
  llvm::Expected<std::string> targetIdentity =
      requireString(**target, "identity", "manifest.target", limits);
  if (!targetIdentity)
    return targetIdentity.takeError();
  llvm::Expected<std::string> runtimeABI =
      requireString(**target, "runtime_abi", "manifest.target", limits);
  if (!runtimeABI)
    return runtimeABI.takeError();
  llvm::Expected<std::string> moduleFormat =
      requireString(**target, "module_format", "manifest.target", limits);
  if (!moduleFormat)
    return moduleFormat.takeError();
  llvm::Expected<TargetIdentityId> parsedTargetIdentity =
      parseTargetIdentityId(*targetIdentity);
  if (!parsedTargetIdentity)
    return parsedTargetIdentity.takeError();
  llvm::Expected<KernelRuntimeABIId> parsedRuntimeABI =
      parseKernelRuntimeABIId(*runtimeABI);
  if (!parsedRuntimeABI)
    return parsedRuntimeABI.takeError();
  llvm::Expected<RuntimeLaunchContract> parsedLaunch =
      parseRuntimeLaunchContract(**launch, "manifest.launch", limits);
  if (!parsedLaunch)
    return parsedLaunch.takeError();
  PackageManifest manifest(*parsedTargetIdentity, *parsedRuntimeABI,
                           std::move(*parsedLaunch), *moduleFormat);
  manifest.program = ProgramId(*programId);
  manifest.cardCount = *cardCount;
  manifest.tileCount = *tileCount;

  llvm::Expected<ProgramDataRecord> parsedProgramData =
      parseProgramDataRecord(**programData, limits);
  if (!parsedProgramData)
    return parsedProgramData.takeError();
  manifest.programData = std::move(*parsedProgramData);

  uint64_t totalRecords = (*programTensors)->size() + (*targetTensors)->size() +
                          (*inputs)->size() + (*outputs)->size() +
                          (*modules)->size() + (*entries)->size();
  if (totalRecords > limits.maxRecords)
    return invalid("package manifest exceeds record limit");
  for (auto [index, value] : llvm::enumerate(**programTensors)) {
    llvm::Expected<ProgramTensorRecord> record =
        parseProgramTensorRecord(value, index, limits);
    if (!record)
      return record.takeError();
    manifest.programTensors.push_back(std::move(*record));
  }
  for (auto [index, value] : llvm::enumerate(**targetTensors)) {
    llvm::Expected<TargetTensorRecord> record =
        parseTargetTensorRecord(value, index, limits);
    if (!record)
      return record.takeError();
    manifest.targetTensors.push_back(std::move(*record));
  }
  for (auto [index, value] : llvm::enumerate(**inputs)) {
    llvm::Expected<ExternalPortRecord> record =
        parseExternalPortRecord(value, index, "inputs", limits);
    if (!record)
      return record.takeError();
    manifest.inputs.push_back(std::move(*record));
  }
  for (auto [index, value] : llvm::enumerate(**outputs)) {
    llvm::Expected<ExternalPortRecord> record =
        parseExternalPortRecord(value, index, "outputs", limits);
    if (!record)
      return record.takeError();
    manifest.outputs.push_back(std::move(*record));
  }
  for (auto [index, value] : llvm::enumerate(**modules)) {
    llvm::Expected<PackageModuleRecord> record =
        parseModuleRecord(value, index, limits);
    if (!record)
      return record.takeError();
    manifest.modules.push_back(std::move(*record));
  }
  for (auto [index, value] : llvm::enumerate(**entries)) {
    llvm::Expected<PackageEntrypointRecord> record =
        parseEntrypointRecord(value, index, limits);
    if (!record)
      return record.takeError();
    manifest.entries.push_back(std::move(*record));
  }
  return manifest;
}

} // namespace wafer::runtime
