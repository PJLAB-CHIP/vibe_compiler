//===- Package.cpp - Typed compiler package assembly ---------------------===//

#include "Wafer/Package/Writer/PackageInternal.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Program/ProgramData.h"
#include "Wafer/Package/Profile/ProfileInstrumentationModel.h"
#include "Wafer/Target/Numeric/Formal/FormalNumeric.h"
#include "Wafer/Target/Numeric/NumericCodec.h"
#include "Wafer/Target/Numeric/NumericSemantics.h"
#include "Wafer/Target/Layout/PhysicalTensorCodec.h"
#include "Wafer/Target/Core/TargetFormat.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace wafer::compiler {
namespace {

llvm::Error fail(llvm::raw_ostream &diagnostics, llvm::StringRef message) {
  diagnostics << "wafer-compile: " << message << "\n";
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

bool pathEntryExists(llvm::StringRef path) {
  llvm::sys::fs::file_type type =
      llvm::sys::fs::get_file_type(path, /*Follow=*/false);
  return type != llvm::sys::fs::file_type::file_not_found &&
         type != llvm::sys::fs::file_type::status_error;
}

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

llvm::Error createDirectory(llvm::StringRef path) {
  if (std::error_code error = llvm::sys::fs::create_directories(path))
    return llvm::createStringError(error, "failed to create package directory");
  return llvm::Error::success();
}

bool haveSameProgramSlice(const frontend::ProgramPartitionSlice &lhs,
                          const frontend::ProgramPartitionSlice &rhs) {
  return lhs.partitionId == rhs.partitionId && lhs.replicaId == rhs.replicaId &&
         lhs.offsets == rhs.offsets && lhs.sizes == rhs.sizes &&
         lhs.strides == rhs.strides && lhs.payloadPath == rhs.payloadPath;
}

bool haveSameProgramBinding(const ProgramResourceBinding &lhs,
                            const ProgramResourceBinding &rhs) {
  return lhs.role == rhs.role && lhs.index == rhs.index &&
         lhs.programIndex == rhs.programIndex && lhs.dtype == rhs.dtype &&
         lhs.distribution == rhs.distribution &&
         lhs.globalShape == rhs.globalShape &&
         lhs.localShape == rhs.localShape &&
         haveSameProgramSlice(lhs.slice, rhs.slice);
}

runtime::PackageAccessMode getAccess(TileEntryArgumentAccess access) {
  switch (access) {
  case TileEntryArgumentAccess::ReadOnly:
    return runtime::PackageAccessMode::ReadOnly;
  case TileEntryArgumentAccess::WriteOnly:
    return runtime::PackageAccessMode::WriteOnly;
  case TileEntryArgumentAccess::ReadWrite:
    return runtime::PackageAccessMode::ReadWrite;
  }
  llvm_unreachable("unknown tile entry argument access");
}

runtime::PackageMemLayout getMemLayout(MemLayout layout) {
  switch (layout) {
  case MemLayout::Tensor:
    return runtime::PackageMemLayout::Tensor;
  case MemLayout::NTensor:
    return runtime::PackageMemLayout::NTensor;
  case MemLayout::Cx:
    return runtime::PackageMemLayout::Cx;
  case MemLayout::NCx:
    return runtime::PackageMemLayout::NCx;
  }
  llvm_unreachable("unknown memory layout");
}

runtime::PackageModuleExportRole getPackageExportRole(TargetExportRole role) {
  switch (role) {
  case TargetExportRole::Prepare:
    return runtime::PackageModuleExportRole::Prepare;
  case TargetExportRole::Main:
    return runtime::PackageModuleExportRole::Main;
  }
  llvm_unreachable("unknown target export role");
}

llvm::StringRef getMainExport(const VerifiedTargetModule &module) {
  for (const VerifiedTargetExport &targetExport : module.getExports())
    if (targetExport.getRole() == TargetExportRole::Main)
      return targetExport.getSymbol();
  return {};
}

struct PortJoin {
  runtime::ExternalPortRecord record;
  std::optional<TileId> seenTile;
};

struct ProgramTensorJoin {
  const ProgramResourceBinding *binding = nullptr;
  const ProgramDataRange *range = nullptr;
  runtime::ProgramTensorRecord record;
};

struct TargetTensorJoin {
  compiler::ProgramTensorId programTensor;
  runtime::TargetTensorRecord record;
  // One (card, tile, launch slot, argument ordinal) consumer per reference.
  std::vector<std::tuple<CardId, TileId, LaunchSlotId, uint64_t>> consumers;
};

llvm::Error checkedAlignUp(uint64_t value, uint64_t alignment,
                           uint64_t &result) {
  if (alignment == 0)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "package placement alignment is zero");
  const uint64_t remainder = value % alignment;
  const uint64_t padding = remainder == 0 ? 0 : alignment - remainder;
  if (padding > std::numeric_limits<uint64_t>::max() - value)
    return llvm::createStringError(llvm::errc::invalid_argument,
                                   "package placement offset overflows");
  result = value + padding;
  return llvm::Error::success();
}

/// Resolves the exact current-target scalar conversion used while producing a
/// selected package representation. A different logical format never falls
/// back to storage-bit reinterpretation. Rounding-mode routes use the model's
/// deterministic nearest-even policy; routes needing a semantic zero-point
/// fail closed because a package descriptor carries no such parameter.
llvm::Expected<std::optional<ResolvedNumericCommand>>
resolvePackageScalarConversion(LogicalFormat source,
                               LogicalFormat destination) {
  if (source == destination)
    return std::optional<ResolvedNumericCommand>{};
  const TargetConvertRoute *route = findTargetConvertRoute(source, destination);
  if (!route)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "current target has no numeric conversion route from %s to %s",
        stringifyLogicalFormat(source).str().c_str(),
        stringifyLogicalFormat(destination).str().c_str());
  std::optional<NumericConvertParameter> parameter;
  switch (route->parameterKind) {
  case TargetConvertParameterKind::None:
    break;
  case TargetConvertParameterKind::RoundingMode:
    parameter =
        NumericConvertParameter::roundingMode(NumericRoundingMode::NearestEven);
    break;
  case TargetConvertParameterKind::ZeroPoint:
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        "target numeric conversion route '%s' requires a zero-point that is "
        "not present in the package tensor descriptor",
        route->canonicalSpelling.str().c_str());
  }
  llvm::Expected<NumericTensorKey> sourceKey =
      NumericTensorKey::create(source, PhysicalTensorLayout::Tensor, {1});
  if (!sourceKey)
    return sourceKey.takeError();
  llvm::Expected<NumericTensorKey> destinationKey =
      NumericTensorKey::create(destination, PhysicalTensorLayout::Tensor, {1});
  if (!destinationKey)
    return destinationKey.takeError();
  llvm::Expected<NumericCommandKey> command =
      NumericCommandKey::createCTConvert(route->opcode, std::move(*sourceKey),
                                         std::move(*destinationKey), parameter);
  if (!command)
    return command.takeError();
  llvm::Expected<ResolvedNumericCommand> resolved = resolveNumericCommand(
      ModelProfileId::formalDeterministic(), std::move(*command));
  if (!resolved)
    return resolved.takeError();
  if (!resolved->isSupported())
    return llvm::createStringError(
        llvm::errc::not_supported,
        "target numeric conversion route '%s' has no implemented formal "
        "semantics",
        route->canonicalSpelling.str().c_str());
  return std::optional<ResolvedNumericCommand>(std::move(*resolved));
}

struct PackageAssembly {
  PackageAssembly(runtime::PackageManifest manifest)
      : manifest(std::move(manifest)) {}

  runtime::PackageManifest manifest;
  /// Owns every joined TargetTensor for the assembly lifetime; `placement`
  /// is a sorted non-owning view into this vector.
  std::vector<std::unique_ptr<TargetTensorJoin>> targetTensorJoins;
  /// Sorted placement order of every package-owned TargetTensor.
  std::vector<TargetTensorJoin *> placement;
};

/// Card-level all-and-only join of the 16 Tile entry arguments with the
/// program boundary bindings and the Q58 owned data ranges. One
/// ProgramTensor is formed per program tensor identity, one TargetTensor per
/// selected target descriptor; entry arguments reference them explicitly.
llvm::Expected<PackageAssembly>
buildManifest(const CardExecutable &cardExecutable,
              const LinkedTargetModules &targetModules,
              llvm::raw_ostream &diagnostics) {
  const ExecutionConfig &config = cardExecutable.getExecutionConfig();
  if (cardExecutable.getRuntimeLaunchContract() !=
      targetModules.getRuntimeLaunchContract())
    return fail(diagnostics,
                "package runtime launch contract does not match executable "
                "and linked target modules");
  if (targetModules.getExecutionConfig() != config ||
      cardExecutable.getTileExecutables().size() !=
          targetModules.getTileInterfaces().size() ||
      cardExecutable.getTileExecutables().size() !=
          static_cast<size_t>(config.getTileCount()))
    return fail(diagnostics,
                "package Tile domain does not match executable and "
                "linked target modules");
  if (targetModules.getModules().empty())
    return fail(diagnostics, "package target module domain is empty");
  const VerifiedTargetModule &firstTargetModule =
      targetModules.getModules().front();
  if (firstTargetModule.getTargetIdentityId() != config.getTargetIdentityId() ||
      firstTargetModule.getKernelRuntimeABIId() !=
          KernelRuntimeABIId::waferTx81Kernel() ||
      firstTargetModule.getModuleFormat() != kCurrentTargetModuleFormat)
    return fail(diagnostics,
                "package target module facts do not match ExecutionConfig "
                "and target registry");
  if (targetModules.getModules().size() != 1)
    return fail(diagnostics, "package target module topology does not match "
                             "runtime launch contract");

  const size_t tileCount = cardExecutable.getTileExecutables().size();
  std::vector<const TileExecutable *> tilesByLaunchSlot(tileCount, nullptr);
  std::vector<const VerifiedTargetTileInterface *> interfacesByLaunchSlot(
      tileCount, nullptr);
  std::set<int64_t> tileIds;
  for (const TileExecutable &tile : cardExecutable.getTileExecutables()) {
    const int64_t launchSlot = tile.getLaunchSlotId().getValue();
    if (tile.getCardId() != CardId(0) || tile.getTileId().getValue() < 0 ||
        launchSlot < 0 || launchSlot >= static_cast<int64_t>(tileCount) ||
        !tileIds.insert(tile.getTileId().getValue()).second ||
        tilesByLaunchSlot[launchSlot] != nullptr)
      return fail(diagnostics,
                  "package executable has invalid or duplicate physical "
                  "Tile identity");
    tilesByLaunchSlot[launchSlot] = &tile;
  }
  for (const VerifiedTargetTileInterface &tileInterface :
       targetModules.getTileInterfaces()) {
    const int64_t launchSlot = tileInterface.getLaunchSlotId().getValue();
    if (tileInterface.getCardId() != CardId(0) ||
        tileInterface.getTileId().getValue() < 0 || launchSlot < 0 ||
        launchSlot >= static_cast<int64_t>(tileCount) ||
        interfacesByLaunchSlot[launchSlot] != nullptr)
      return fail(diagnostics,
                  "package target module has invalid or duplicate physical "
                  "Tile interface identity");
    interfacesByLaunchSlot[launchSlot] = &tileInterface;
  }
  if (llvm::is_contained(tilesByLaunchSlot, nullptr) ||
      llvm::is_contained(interfacesByLaunchSlot, nullptr))
    return fail(diagnostics,
                "package launch-slot domain is not dense and complete");

  const std::vector<ProgramResourceBinding> &sharedProgramBindings =
      tilesByLaunchSlot.front()->getProgramBindings();
  for (const TileExecutable *tile : tilesByLaunchSlot) {
    const std::vector<ProgramResourceBinding> &bindings =
        tile->getProgramBindings();
    if (bindings.size() != sharedProgramBindings.size() ||
        !std::equal(bindings.begin(), bindings.end(),
                    sharedProgramBindings.begin(), haveSameProgramBinding))
      return fail(diagnostics, "Tile module boundaries do not agree on the "
                               "typed card-shared resource domain");
  }

  PackageAssembly assembly(
      runtime::PackageManifest(firstTargetModule.getTargetIdentityId(),
                               firstTargetModule.getKernelRuntimeABIId(),
                               targetModules.getRuntimeLaunchContract(),
                               firstTargetModule.getModuleFormat()));
  runtime::PackageManifest &manifest = assembly.manifest;
  manifest.program = runtime::ProgramId(0);
  manifest.cardCount = 1;
  manifest.tileCount = config.getTileCount();

  for (auto [expectedModuleId, target] :
       llvm::enumerate(targetModules.getModules())) {
    if (target.getId().getValue() != expectedModuleId ||
        target.getTargetIdentityId() !=
            firstTargetModule.getTargetIdentityId() ||
        target.getKernelRuntimeABIId() !=
            firstTargetModule.getKernelRuntimeABIId() ||
        target.getModuleFormat() != firstTargetModule.getModuleFormat())
      return fail(diagnostics, "package target module domain is not canonical");
    bool seenPrepare = false;
    bool seenMain = false;
    runtime::PackageModuleRecord module;
    module.id = runtime::ModuleId(expectedModuleId);
    module.relativePath = target.getRelativePath().str();
    module.digest = target.getContentDigest().str();
    module.format = target.getModuleFormat().str();
    for (const VerifiedTargetExport &targetExport : target.getExports()) {
      bool *seen = targetExport.getRole() == TargetExportRole::Prepare
                       ? &seenPrepare
                       : &seenMain;
      if (*seen || targetExport.getSymbol().empty())
        return fail(diagnostics,
                    "package target module exports are not canonical");
      *seen = true;
      module.exports.push_back({getPackageExportRole(targetExport.getRole()),
                                targetExport.getSymbol().str()});
    }
    const bool hasPrepare = llvm::is_contained(manifest.launch.getPhases(),
                                               RuntimeLaunchPhaseRole::Prepare);
    if (!seenMain || seenPrepare != hasPrepare ||
        module.exports.size() != (hasPrepare ? 2u : 1u))
      return fail(
          diagnostics,
          "package target module exports do not match runtime launch contract");
    manifest.modules.push_back(std::move(module));
  }

  const ProgramDataHandoff &handoff = cardExecutable.getProgramDataHandoff();

  // Join state.
  std::map<ProgramTensorId, std::unique_ptr<ProgramTensorJoin>>
      programTensorJoins;
  std::vector<std::unique_ptr<TargetTensorJoin>> targetTensorJoins;
  std::map<int64_t, PortJoin> inputJoins;  // keyed by programIndex
  std::map<int64_t, PortJoin> outputJoins; // keyed by programIndex

  auto getBinding =
      [&](const TileExecutable &tile, TileEntryArgumentKind kind,
          int64_t resourceIndex) -> const ProgramResourceBinding * {
    if (kind != TileEntryArgumentKind::ExternalInput &&
        kind != TileEntryArgumentKind::TargetTensor &&
        kind != TileEntryArgumentKind::ExternalOutput)
      return nullptr;
    const ProgramResourceBinding *match = nullptr;
    for (const ProgramResourceBinding &candidate : tile.getProgramBindings()) {
      bool roleMatches = false;
      switch (kind) {
      case TileEntryArgumentKind::ExternalInput:
        roleMatches = candidate.role == ProgramResourceRole::UserInput;
        break;
      case TileEntryArgumentKind::TargetTensor:
        roleMatches = candidate.role == ProgramResourceRole::Parameter ||
                      candidate.role == ProgramResourceRole::Constant;
        break;
      case TileEntryArgumentKind::ExternalOutput:
        roleMatches = candidate.role == ProgramResourceRole::Output;
        break;
      default:
        break;
      }
      if (!roleMatches || candidate.index != resourceIndex)
        continue;
      if (match)
        return nullptr; // caller reports the duplicate
      match = &candidate;
    }
    return match;
  };

  for (int64_t launchSlot = 0; launchSlot < static_cast<int64_t>(tileCount);
       ++launchSlot) {
    const TileExecutable &tile = *tilesByLaunchSlot[launchSlot];
    const VerifiedTargetTileInterface &tileInterface =
        *interfacesByLaunchSlot[launchSlot];
    const uint64_t moduleId = tileInterface.getModuleId().getValue();
    if (moduleId >= targetModules.getModules().size())
      return fail(diagnostics,
                  "package Tile launch interface references an unknown "
                  "module");
    const VerifiedTargetModule &target = targetModules.getModules()[moduleId];
    if (tile.getCardId() != CardId(0) ||
        tile.getCardId() != tileInterface.getCardId() ||
        tile.getTileId() != tileInterface.getTileId() ||
        tile.getLaunchSlotId() != tileInterface.getLaunchSlotId() ||
        target.getId() != tileInterface.getModuleId() || moduleId != 0 ||
        tile.getEntrySymbol() != getMainExport(target))
      return fail(diagnostics,
                  "package Tile/module/entry/profile domain is not canonical");

    runtime::PackageEntrypointRecord entry;
    entry.id = runtime::EntryId(static_cast<uint64_t>(launchSlot));
    entry.cardId = tile.getCardId();
    entry.tileId = tile.getTileId();
    entry.launchSlot = runtime::LaunchSlotId(
        static_cast<uint64_t>(tile.getLaunchSlotId().getValue()));
    entry.module = runtime::ModuleId(moduleId);
    entry.completion =
        runtime::PackageEntryCompletionKind::ReturnAfterLocalDrain;

    for (const TileEntryArgument &slot :
         tileInterface.getTileEntryArguments()) {
      if (slot.ordinal != static_cast<int64_t>(entry.arguments.size()) ||
          slot.byteSize <= 0 || slot.alignment <= 0)
        return fail(diagnostics,
                    "package input tile entry arguments are not canonical");

      if (slot.kind == TileEntryArgumentKind::ExternalInput ||
          slot.kind == TileEntryArgumentKind::ExternalOutput) {
        const ProgramResourceBinding *binding =
            getBinding(tile, slot.kind, slot.resourceIndex);
        if (!binding)
          return fail(diagnostics,
                      "package tile entry argument does not match executable "
                      "resource binding");
        std::map<int64_t, PortJoin> &joins =
            slot.kind == TileEntryArgumentKind::ExternalInput ? inputJoins
                                                              : outputJoins;
        auto existing = joins.find(binding->programIndex);
        if (existing == joins.end()) {
          PortJoin join;
          join.record.id = runtime::PortId(static_cast<uint64_t>(joins.size()));
          join.record.roleIndex = binding->programIndex;
          join.record.logicalDtype = binding->dtype;
          join.record.logicalShape = binding->localShape;
          join.record.dtype = slot.dtype;
          join.record.layout = getMemLayout(slot.layout);
          join.record.shape = slot.shape;
          join.record.bytes = static_cast<uint64_t>(slot.byteSize);
          join.record.alignment = static_cast<uint64_t>(slot.alignment);
          join.seenTile = tile.getTileId();
          joins.emplace(binding->programIndex, std::move(join));
        } else {
          const runtime::ExternalPortRecord &record = existing->second.record;
          if (record.logicalDtype != binding->dtype ||
              record.logicalShape != binding->localShape ||
              record.dtype != slot.dtype ||
              record.layout != getMemLayout(slot.layout) ||
              record.shape != slot.shape ||
              record.bytes != static_cast<uint64_t>(slot.byteSize) ||
              record.alignment != static_cast<uint64_t>(slot.alignment))
            return fail(diagnostics,
                        "Tile entry arguments disagree on one external port");
          existing->second.seenTile = tile.getTileId();
        }
        entry.arguments.push_back(
            {static_cast<uint64_t>(slot.ordinal),
             slot.kind == TileEntryArgumentKind::ExternalInput
                 ? runtime::TileEntryArgumentReference(
                       runtime::ExternalInputArgument{
                           joins.at(binding->programIndex).record.id})
                 : runtime::TileEntryArgumentReference(
                       runtime::ExternalOutputArgument{
                           joins.at(binding->programIndex).record.id}),
             getAccess(slot.access)});
        continue;
      }

      if (slot.kind == TileEntryArgumentKind::TargetTensor) {
        const ProgramResourceBinding *binding =
            getBinding(tile, slot.kind, slot.resourceIndex);
        if (!binding)
          return fail(diagnostics,
                      "package tile entry argument does not match executable "
                      "resource binding");
        ProgramTensorJoin *programTensor = nullptr;
        auto programExisting =
            programTensorJoins.find(binding->programTensorId);
        if (programExisting == programTensorJoins.end()) {
          auto join = std::make_unique<ProgramTensorJoin>();
          join->binding = binding;
          const ProgramDataRange *range =
              handoff.findRange(binding->programTensorId);
          if (!range)
            return fail(diagnostics, "program tensor has no owned data range");
          if (range->getDType() != binding->dtype ||
              !(range->getLocalShape() ==
                llvm::ArrayRef<int64_t>(binding->localShape)) ||
              !(range->getGlobalShape() ==
                llvm::ArrayRef<int64_t>(binding->globalShape)) ||
              !(range->getSliceOffsets() ==
                llvm::ArrayRef<int64_t>(binding->slice.offsets)) ||
              !(range->getSliceSizes() ==
                llvm::ArrayRef<int64_t>(binding->slice.sizes)))
            return fail(diagnostics,
                        "program data range disagrees with typed Tile "
                        "binding");
          join->range = range;
          join->record.id = runtime::ProgramTensorId(
              static_cast<uint64_t>(programTensorJoins.size()));
          join->record.role = binding->role == ProgramResourceRole::Parameter
                                  ? runtime::ProgramTensorRole::Parameter
                                  : runtime::ProgramTensorRole::Constant;
          join->record.roleIndex = binding->programTensorId.roleIndex;
          join->record.dtype = binding->dtype;
          join->record.globalShape = binding->globalShape;
          join->record.localShape = binding->localShape;
          join->record.sliceOffsets = binding->slice.offsets;
          join->record.sliceSizes = binding->slice.sizes;
          programTensor = join.get();
          programTensorJoins.emplace(binding->programTensorId, std::move(join));
        } else {
          programTensor = programExisting->second.get();
          if (programTensor->record.dtype != binding->dtype ||
              programTensor->record.globalShape != binding->globalShape ||
              programTensor->record.localShape != binding->localShape ||
              programTensor->record.sliceOffsets != binding->slice.offsets ||
              programTensor->record.sliceSizes != binding->slice.sizes)
            return fail(diagnostics,
                        "Tile bindings disagree on one program tensor");
        }
        const runtime::ProgramTensorId programTensorId =
            programTensor->record.id;
        const runtime::PackageMemLayout layout = getMemLayout(slot.layout);
        const auto match = [&](const TargetTensorJoin &join) {
          return join.record.programTensor == programTensorId &&
                 join.record.dtype == slot.dtype &&
                 join.record.layout == layout &&
                 join.record.shape == slot.shape &&
                 join.record.bytes == static_cast<uint64_t>(slot.byteSize) &&
                 join.record.alignment == static_cast<uint64_t>(slot.alignment);
        };
        auto targetMatch = llvm::find_if(
            targetTensorJoins, [&](const auto &join) { return match(*join); });
        TargetTensorJoin *targetTensor = nullptr;
        if (targetMatch == targetTensorJoins.end()) {
          auto join = std::make_unique<TargetTensorJoin>();
          join->programTensor = binding->programTensorId;
          join->record.id = runtime::TargetTensorId(
              static_cast<uint64_t>(targetTensorJoins.size()));
          join->record.programTensor = programTensorId;
          join->record.dtype = slot.dtype;
          join->record.layout = layout;
          join->record.shape = slot.shape;
          join->record.bytes = static_cast<uint64_t>(slot.byteSize);
          join->record.alignment = static_cast<uint64_t>(slot.alignment);
          join->record.fileOffset = 0;
          targetTensor = join.get();
          targetTensorJoins.push_back(std::move(join));
        } else {
          targetTensor = targetMatch->get();
        }
        targetTensor->consumers.push_back(
            {tile.getCardId(), tile.getTileId(), tile.getLaunchSlotId(),
             static_cast<uint64_t>(slot.ordinal)});
        entry.arguments.push_back(
            {static_cast<uint64_t>(slot.ordinal),
             runtime::TileEntryArgumentReference(
                 runtime::TargetTensorArgument{targetTensor->record.id}),
             getAccess(slot.access)});
        continue;
      }

      if (slot.kind == TileEntryArgumentKind::Workspace) {
        if (!detail::isValidPackageCompilerManagedSlot(slot))
          return fail(diagnostics,
                      "package workspace tile entry argument identity is "
                      "invalid");
        entry.arguments.push_back(
            {static_cast<uint64_t>(slot.ordinal),
             runtime::TileEntryArgumentReference(runtime::WorkspaceArgument{
                 static_cast<uint64_t>(slot.byteSize),
                 static_cast<uint64_t>(slot.alignment)}),
             getAccess(slot.access)});
        continue;
      }

      if (slot.kind == TileEntryArgumentKind::ProfileRecord) {
        if (!detail::isValidPackageCompilerManagedSlot(slot))
          return fail(diagnostics,
                      "package profile record tile entry argument identity "
                      "is invalid");
        entry.arguments.push_back(
            {static_cast<uint64_t>(slot.ordinal),
             runtime::TileEntryArgumentReference(runtime::ProfileRecordArgument{
                 runtime::kProfileRecordABI.str(),
                 static_cast<uint64_t>(slot.byteSize),
                 static_cast<uint64_t>(slot.alignment)}),
             getAccess(slot.access)});
        continue;
      }

      if (slot.kind == TileEntryArgumentKind::TransportStatus) {
        if (!detail::isValidPackageCompilerManagedSlot(slot))
          return fail(diagnostics,
                      "package Direct DTE status tile entry argument "
                      "identity is invalid");
        entry.arguments.push_back(
            {static_cast<uint64_t>(slot.ordinal),
             runtime::TileEntryArgumentReference(
                 runtime::TransportStatusArgument{
                     runtime::kDirectDTEStatusABI.str(),
                     runtime::kDirectDTEStatusStorageBytes,
                     runtime::kDirectDTEStatusStorageAlignment}),
             getAccess(slot.access)});
        continue;
      }
      return fail(diagnostics,
                  "package tile entry argument has an unknown kind");
    }

    if (tile.getTransportContract() == TransportContract::DirectDTE) {
      entry.transport = runtime::DirectDTETransportRequirements{
          runtime::kDirectDTEStatusABI.str(), true};
    } else {
      entry.transport = runtime::NoTransportRequirements{};
    }
    manifest.entries.push_back(std::move(entry));
  }

  // All-and-only coverage: every joined program tensor must have at least
  // one TargetTensor consumer. Port and argument coverage across the 16
  // Tiles is re-verified by the canonical manifest verification.
  llvm::SmallDenseSet<uint64_t> consumedProgramTensors;
  for (const auto &join : targetTensorJoins)
    consumedProgramTensors.insert(join->record.programTensor.getValue());
  for (const auto &pair : programTensorJoins) {
    if (!consumedProgramTensors.contains(pair.second->record.id.getValue()))
      return fail(diagnostics,
                  "package program tensor has no TargetTensor consumer");
  }

  // Emit tables in deterministic join order.
  for (const auto &pair : inputJoins)
    manifest.inputs.push_back(pair.second.record);
  for (const auto &pair : outputJoins)
    manifest.outputs.push_back(pair.second.record);
  for (const auto &pair : programTensorJoins)
    manifest.programTensors.push_back(pair.second->record);

  // Deterministic placement by stable identity: program tensor id first,
  // then the complete target descriptor as tie-break.
  assembly.targetTensorJoins = std::move(targetTensorJoins);
  assembly.placement.reserve(assembly.targetTensorJoins.size());
  for (const auto &join : assembly.targetTensorJoins)
    assembly.placement.push_back(join.get());
  llvm::sort(assembly.placement,
             [](TargetTensorJoin *lhs, TargetTensorJoin *rhs) {
               const runtime::TargetTensorRecord &left = lhs->record;
               const runtime::TargetTensorRecord &right = rhs->record;
               return std::tie(left.programTensor, left.dtype, left.layout,
                               left.shape, left.bytes, left.alignment) <
                      std::tie(right.programTensor, right.dtype, right.layout,
                               right.shape, right.bytes, right.alignment);
             });
  uint64_t nextOffset = 0;
  uint64_t baseAlignment = 1;
  // Entries were recorded under discovery-order TargetTensor ids; re-issue
  // canonical sorted ids and rewrite the typed entry references in place.
  std::vector<runtime::TargetTensorId> canonicalTargetTensorId(
      assembly.targetTensorJoins.size());
  for (auto [index, join] : llvm::enumerate(assembly.placement)) {
    runtime::TargetTensorRecord &record = join->record;
    canonicalTargetTensorId[record.id.getValue()] =
        runtime::TargetTensorId(static_cast<uint64_t>(index));
    record.id = runtime::TargetTensorId(static_cast<uint64_t>(index));
    uint64_t offset = 0;
    if (llvm::Error error =
            checkedAlignUp(nextOffset, record.alignment, offset))
      return fail(diagnostics, llvm::toString(std::move(error)));
    if (record.bytes > std::numeric_limits<uint64_t>::max() - offset)
      return fail(diagnostics, "package program data layout overflows");
    record.fileOffset = offset;
    nextOffset = offset + record.bytes;
    baseAlignment = std::max(baseAlignment, record.alignment);
  }
  for (runtime::PackageEntrypointRecord &entry : manifest.entries) {
    for (runtime::TileEntryArgumentRecord &argument : entry.arguments) {
      if (const runtime::TargetTensorArgument *target =
              std::get_if<runtime::TargetTensorArgument>(&argument.reference)) {
        argument.reference = runtime::TargetTensorArgument{
            canonicalTargetTensorId[target->tensor.getValue()]};
      }
    }
  }
  manifest.targetTensors.reserve(assembly.placement.size());
  for (TargetTensorJoin *join : assembly.placement)
    manifest.targetTensors.push_back(join->record);
  manifest.programData.totalBytes = nextOffset;
  manifest.programData.baseAlignment = baseAlignment;
  return assembly;
}

llvm::Error copyTargetModules(const LinkedTargetModules &targetModules,
                              llvm::StringRef stagingRoot,
                              llvm::raw_ostream &diagnostics,
                              std::optional<int64_t> failAfterLaunchSlot) {
  for (const VerifiedTargetModule &module : targetModules.getModules()) {
    llvm::SmallString<256> source(targetModules.getRootDirectory());
    llvm::sys::path::append(source, module.getRelativePath());
    if (!isRegularFile(source))
      return fail(diagnostics,
                  "verified target module disappeared before package "
                  "assembly");
    llvm::SmallString<256> destination(stagingRoot);
    llvm::sys::path::append(destination, module.getRelativePath());
    llvm::SmallString<256> parent(destination);
    llvm::sys::path::remove_filename(parent);
    if (llvm::Error error = createDirectory(parent))
      return fail(diagnostics, llvm::toString(std::move(error)));
    if (std::error_code error = llvm::sys::fs::copy_file(source, destination))
      return fail(diagnostics,
                  "failed to copy verified target module: " + error.message());
  }
  if (failAfterLaunchSlot && *failAfterLaunchSlot >= 0 &&
      *failAfterLaunchSlot <
          static_cast<int64_t>(targetModules.getTileInterfaces().size()))
    return fail(diagnostics,
                "test-only injected package failure after launch slot " +
                    std::to_string(*failAfterLaunchSlot));
  return llvm::Error::success();
}

/// Streams one materialized TargetTensor into the program-data staging file
/// at its placement offset with canonical zero padding, updating the whole
/// file digest. Each selected target representation is materialized exactly
/// once; the identity conversion requires the target descriptor to equal the
/// owned source range descriptor.
llvm::Expected<std::string> writeProgramData(llvm::StringRef programDataPath,
                                             PackageAssembly &assembly,
                                             const ProgramDataHandoff &handoff,
                                             llvm::raw_ostream &diagnostics) {
  std::error_code error;
  llvm::raw_fd_ostream output(programDataPath, error, llvm::sys::fs::OF_None);
  if (error)
    return fail(diagnostics,
                "failed to create package program data: " + error.message());
  llvm::SHA256 hasher;

  constexpr uint64_t kWindowBytes = kProgramDataReadWindowBytes;

  auto writeWindow = [&](const uint8_t *data, uint64_t bytes) -> llvm::Error {
    uint64_t written = 0;
    while (written < bytes) {
      const uint64_t chunk = std::min<uint64_t>(kWindowBytes, bytes - written);
      output.write(reinterpret_cast<const char *>(data + written), chunk);
      if (output.has_error())
        return fail(diagnostics, "failed to write package program data");
      hasher.update(
          llvm::StringRef(reinterpret_cast<const char *>(data + written),
                          static_cast<size_t>(chunk)));
      written += chunk;
    }
    return llvm::Error::success();
  };

  auto writeZeros = [&](uint64_t bytes) -> llvm::Error {
    constexpr uint64_t kWindowBytes = kProgramDataReadWindowBytes;
    std::vector<uint8_t> zeros(static_cast<size_t>(kWindowBytes), 0);
    uint64_t remaining = bytes;
    while (remaining > 0) {
      const uint64_t chunk = std::min(remaining, kWindowBytes);
      output.write(reinterpret_cast<const char *>(zeros.data()), chunk);
      if (output.has_error())
        return fail(diagnostics,
                    "failed to write package program data padding");
      hasher.update(
          llvm::StringRef(reinterpret_cast<const char *>(zeros.data()),
                          static_cast<size_t>(chunk)));
      remaining -= chunk;
    }
    return llvm::Error::success();
  };

  const ModelProfileRecord &model =
      getModelProfileRecord(ModelProfileId::formalDeterministic());
  const LogicalScalarCodecPolicy decodePolicy = model.numericDecodePolicy;
  const LogicalScalarCodecPolicy encodePolicy = model.numericEncodePolicy;

  uint64_t cursor = 0;
  for (TargetTensorJoin *join : assembly.placement) {
    const runtime::TargetTensorRecord &record = join->record;
    if (record.fileOffset > cursor) {
      if (llvm::Error paddingError = writeZeros(record.fileOffset - cursor))
        return paddingError;
      cursor = record.fileOffset;
    }
    const ProgramDataRange *range = handoff.findRange(join->programTensor);
    if (!range)
      return fail(
          diagnostics,
          "package TargetTensor has no owned data range (role=" +
              std::to_string(static_cast<int>(join->programTensor.role)) +
              " index=" + std::to_string(join->programTensor.roleIndex) + ")");
    llvm::Expected<LogicalFormat> targetFormat =
        parseLogicalFormat(record.dtype);
    if (!targetFormat)
      return fail(diagnostics, "package TargetTensor has an unknown target "
                               "dtype '" +
                                   record.dtype + "'");
    llvm::Expected<LogicalFormat> sourceFormat =
        parseLogicalFormat(range->getDType());
    if (!sourceFormat)
      return fail(diagnostics, "package TargetTensor has an unknown source "
                               "dtype '" +
                                   range->getDType().str() + "'");
    const LogicalFormatDescriptor *targetDescriptor =
        findLogicalFormatDescriptor(*targetFormat);
    const LogicalFormatDescriptor *sourceDescriptor =
        findLogicalFormatDescriptor(*sourceFormat);
    if (!targetDescriptor || !sourceDescriptor)
      return fail(diagnostics, "package TargetTensor format is not "
                               "registered in the target format table");
    const PhysicalTensorLayout physicalLayout =
        runtime::getPhysicalTensorLayout(record.layout);
    std::vector<uint64_t> targetShape(record.shape.begin(), record.shape.end());
    llvm::Expected<NumericTensorKey> targetKey =
        NumericTensorKey::create(*targetFormat, physicalLayout, targetShape);
    if (!targetKey)
      return fail(diagnostics, "package TargetTensor physical layout is "
                               "invalid: " +
                                   llvm::toString(targetKey.takeError()));
    llvm::Expected<uint64_t> storageBytes =
        getPhysicalTensorStorageBytes(*targetKey);
    if (!storageBytes)
      return fail(diagnostics, "package TargetTensor physical layout is "
                               "invalid: " +
                                   llvm::toString(storageBytes.takeError()));
    if (*storageBytes != record.bytes)
      return fail(diagnostics,
                  "package TargetTensor bytes disagree with the shared "
                  "physical tensor codec (codec=" +
                      std::to_string(*storageBytes) +
                      " manifest=" + std::to_string(record.bytes) + ")");
    if (sourceDescriptor->storageBits == 1 ||
        sourceDescriptor->storageBits % 8 != 0)
      return fail(diagnostics,
                  "bit-packed source program tensors are not admitted by "
                  "the current program-data boundary");
    const uint64_t sourceElementBytes =
        static_cast<uint64_t>(sourceDescriptor->storageBits / 8);
    if (range->getRegionLength() % sourceElementBytes != 0)
      return fail(diagnostics,
                  "package TargetTensor source region is not element-aligned");
    const uint64_t sourceElementCount =
        range->getRegionLength() / sourceElementBytes;

    const bool identityLayout =
        record.layout == runtime::PackageMemLayout::Tensor ||
        record.layout == runtime::PackageMemLayout::NTensor;
    if (*targetFormat == *sourceFormat && identityLayout &&
        record.bytes == range->getRegionLength()) {
      // Identity representation: the target bytes are the source region
      // bytes; stream them with bounded windows and no host copy.
      llvm::Expected<ProgramDataRangeMaterialization> materialization =
          handoff.beginRangeMaterialization(join->programTensor);
      if (!materialization)
        return fail(diagnostics,
                    "package TargetTensor could not begin source range "
                    "materialization: " +
                        llvm::toString(materialization.takeError()));
      uint64_t written = 0;
      while (written < record.bytes) {
        const uint64_t chunk =
            std::min<uint64_t>(kWindowBytes, record.bytes - written);
        std::vector<uint8_t> window(static_cast<size_t>(chunk));
        if (llvm::Error materializeError =
                materialization->materializeWindow(written, window))
          return fail(diagnostics, llvm::toString(std::move(materializeError)));
        if (llvm::Error writeError = writeWindow(window.data(), chunk))
          return writeError;
        written += chunk;
      }
      cursor += record.bytes;
      continue;
    }

    // Selected representation: convert the source logical values exactly
    // once through the shared codec using bounded value windows.
    if (targetKey->getElementCount() != sourceElementCount)
      return fail(diagnostics,
                  "package TargetTensor element count disagrees with the "
                  "source program tensor region");
    llvm::Expected<std::optional<ResolvedNumericCommand>> conversion =
        resolvePackageScalarConversion(*sourceFormat, *targetFormat);
    if (!conversion)
      return fail(diagnostics, "package TargetTensor numeric conversion is "
                               "unsupported: " +
                                   llvm::toString(conversion.takeError()));
    llvm::Expected<PhysicalTensorWindowPlan> physicalPlan =
        PhysicalTensorWindowPlan::create(*targetKey);
    if (!physicalPlan)
      return fail(diagnostics, "package TargetTensor physical window planning "
                               "failed: " +
                                   llvm::toString(physicalPlan.takeError()));
    const uint64_t valueBudget =
        std::max<uint64_t>(1, kWindowBytes / sourceElementBytes);
    llvm::Expected<ProgramDataRangeMaterialization> materialization =
        handoff.beginRangeMaterialization(join->programTensor);
    if (!materialization)
      return fail(diagnostics,
                  "package TargetTensor could not begin source range "
                  "materialization: " +
                      llvm::toString(materialization.takeError()));
    while (!physicalPlan->done()) {
      llvm::Expected<PhysicalTensorWindowPlan::WriteWindow> window =
          physicalPlan->takeNext(kWindowBytes, valueBudget,
                                 /*paddingFill=*/0);
      if (!window)
        return fail(diagnostics, "package TargetTensor physical window "
                                 "planning failed: " +
                                     llvm::toString(window.takeError()));

      // Physical Cx/NCx order can visit non-contiguous logical values. Sort
      // only this bounded window's mappings and materialize consecutive
      // source runs; no full logical or physical tensor is resident on host.
      std::vector<size_t> logicalOrder(window->elements.size());
      std::iota(logicalOrder.begin(), logicalOrder.end(), 0);
      llvm::sort(logicalOrder, [&](size_t lhs, size_t rhs) {
        return window->elements[lhs].logicalIndex <
               window->elements[rhs].logicalIndex;
      });
      size_t orderIndex = 0;
      while (orderIndex < logicalOrder.size()) {
        const uint64_t firstLogical =
            window->elements[logicalOrder[orderIndex]].logicalIndex;
        size_t runEnd = orderIndex + 1;
        while (runEnd < logicalOrder.size() &&
               window->elements[logicalOrder[runEnd]].logicalIndex ==
                   firstLogical + (runEnd - orderIndex))
          ++runEnd;
        const uint64_t runValues = runEnd - orderIndex;
        std::vector<uint8_t> sourceWindow(
            static_cast<size_t>(runValues * sourceElementBytes));
        if (llvm::Error materializeError = materialization->materializeWindow(
                firstLogical * sourceElementBytes, sourceWindow))
          return fail(diagnostics, llvm::toString(std::move(materializeError)));
        for (size_t current = orderIndex; current < runEnd; ++current) {
          const auto &element = window->elements[logicalOrder[current]];
          const uint64_t runIndex = element.logicalIndex - firstLogical;
          llvm::Expected<RawLogicalValue> value = readRawLogicalValue(
              *sourceFormat, sourceWindow,
              runIndex * static_cast<uint64_t>(sourceDescriptor->storageBits),
              decodePolicy);
          if (!value)
            return fail(diagnostics,
                        "package TargetTensor source value read failed: " +
                            llvm::toString(value.takeError()));
          RawLogicalValue targetValue = *value;
          if (*conversion) {
            llvm::Expected<FormalNumericResult> converted =
                evaluateFormalConvert(**conversion, *value);
            if (!converted)
              return fail(diagnostics,
                          "package TargetTensor numeric conversion failed: " +
                              llvm::toString(converted.takeError()));
            targetValue = converted->value;
          }
          if (llvm::Error encodeError =
                  writeRawLogicalValue(targetValue, window->bytes,
                                       element.windowBitOffset, encodePolicy))
            return fail(diagnostics,
                        "package TargetTensor target value write failed: " +
                            llvm::toString(std::move(encodeError)));
        }
        orderIndex = runEnd;
      }
      const uint64_t windowFileOffset =
          record.fileOffset + window->physicalOffset;
      if (windowFileOffset > cursor) {
        if (llvm::Error paddingError = writeZeros(windowFileOffset - cursor))
          return paddingError;
        cursor = windowFileOffset;
      } else if (windowFileOffset < cursor) {
        return fail(diagnostics,
                    "package TargetTensor window overlap is invalid");
      }
      if (llvm::Error writeError =
              writeWindow(window->bytes.data(), window->bytes.size()))
        return writeError;
      cursor += window->bytes.size();
    }
    if (physicalPlan->getPlannedValueCount() != sourceElementCount)
      return fail(diagnostics, "package TargetTensor conversion is incomplete");
  }
  if (assembly.manifest.programData.totalBytes != cursor)
    return fail(diagnostics,
                "package program data does not end at the last canonical "
                "TargetTensor range");
  output.close();
  if (output.has_error())
    return fail(diagnostics, "failed to write package program data");
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

llvm::Error writeManifest(llvm::StringRef stagingRoot,
                          const runtime::VerifiedPackageManifest &manifest) {
  llvm::SmallString<256> path(stagingRoot);
  llvm::sys::path::append(path, runtime::kPackageManifestFileName);
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::createStringError(error, "failed to open package manifest");
  output << runtime::serializeCanonicalPackageJson(manifest);
  output.close();
  if (output.has_error())
    return llvm::createStringError(llvm::errc::io_error,
                                   "failed to write package manifest");
  return llvm::Error::success();
}

llvm::Error fsyncPath(llvm::StringRef path, bool directory) {
#ifdef __linux__
  std::string storage = path.str();
  int flags = O_RDONLY | O_CLOEXEC;
  if (directory)
    flags |= O_DIRECTORY;
  int descriptor = ::open(storage.c_str(), flags);
  if (descriptor < 0)
    return llvm::createStringError(
        std::error_code(errno, std::generic_category()),
        "failed to open package member for fsync");
  int result = ::fsync(descriptor);
  int errorNumber = errno;
  ::close(descriptor);
  if (result != 0)
    return llvm::createStringError(
        std::error_code(errorNumber, std::generic_category()),
        "failed to fsync package member");
  return llvm::Error::success();
#else
  (void)path;
  (void)directory;
  return llvm::createStringError(llvm::errc::not_supported,
                                 "package fsync is unsupported on this host");
#endif
}

llvm::Error fsyncPackageTree(llvm::StringRef root) {
  std::vector<std::string> directories;
  directories.push_back(root.str());
  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(root, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return llvm::createStringError(error, "failed to walk package for fsync");
    if (iterator->type() == llvm::sys::fs::file_type::directory_file) {
      directories.push_back(iterator->path());
      continue;
    }
    if (iterator->type() != llvm::sys::fs::file_type::regular_file)
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "package contains unsupported member during fsync");
    if (llvm::Error syncError = fsyncPath(iterator->path(), false))
      return syncError;
  }
  if (error)
    return llvm::createStringError(error, "failed to walk package for fsync");
  for (llvm::StringRef directory : llvm::reverse(directories))
    if (llvm::Error syncError = fsyncPath(directory, true))
      return syncError;
  return llvm::Error::success();
}

bool renameDirectoryNoReplace(llvm::StringRef source,
                              llvm::StringRef destination,
                              llvm::raw_ostream &diagnostics) {
#ifdef __linux__
  std::string sourceStorage = source.str();
  std::string destinationStorage = destination.str();
  if (::syscall(SYS_renameat2, AT_FDCWD, sourceStorage.c_str(), AT_FDCWD,
                destinationStorage.c_str(), RENAME_NOREPLACE) == 0)
    return false;
  diagnostics << "wafer-compile: package_write_failed: "
              << std::error_code(errno, std::generic_category()).message()
              << "\n";
  return true;
#else
  (void)source;
  (void)destination;
  diagnostics << "wafer-compile: package_write_failed: no-replace directory "
                 "rename is unsupported on this host\n";
  return true;
#endif
}

} // namespace

llvm::Expected<runtime::VerifiedPackageManifest>
detail::writePackage(llvm::StringRef tensorProgramDirectory,
                     const CardExecutable &cardExecutable,
                     const LinkedTargetModules &targetModules,
                     llvm::StringRef outputDirectory,
                     llvm::raw_ostream &diagnostics,
                     std::optional<int64_t> failAfterLaunchSlot) {
  (void)tensorProgramDirectory;
  if (outputDirectory.empty())
    return fail(diagnostics,
                "package input/output directory must not be empty");
  if (pathEntryExists(outputDirectory))
    return fail(diagnostics, "refusing to replace existing package directory");

  llvm::SmallString<256> outputParent(outputDirectory);
  llvm::sys::path::remove_filename(outputParent);
  if (outputParent.empty())
    outputParent = ".";
  if (llvm::Error error = createDirectory(outputParent))
    return fail(diagnostics, llvm::toString(std::move(error)));
  llvm::SmallString<256> stagingPrefix(outputParent);
  llvm::sys::path::append(stagingPrefix, ".wafer-package-staging");
  llvm::SmallString<256> stagingRoot;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(stagingPrefix, stagingRoot))
    return fail(diagnostics,
                "failed to create package staging: " + error.message());
  auto cleanup = llvm::make_scope_exit(
      [&] { llvm::sys::fs::remove_directories(stagingRoot); });

  if (llvm::Error error = copyTargetModules(targetModules, stagingRoot,
                                            diagnostics, failAfterLaunchSlot))
    return std::move(error);

  llvm::Expected<PackageAssembly> assembly =
      buildManifest(cardExecutable, targetModules, diagnostics);
  if (!assembly)
    return assembly.takeError();

  llvm::SmallString<256> dataDirectory(stagingRoot);
  llvm::sys::path::append(dataDirectory, "data");
  if (llvm::Error error = createDirectory(dataDirectory))
    return fail(diagnostics, llvm::toString(std::move(error)));
  llvm::SmallString<256> programDataPath(dataDirectory);
  llvm::sys::path::append(programDataPath, "program-data.bin");
  llvm::Expected<std::string> programDataDigest =
      writeProgramData(programDataPath, *assembly,
                       cardExecutable.getProgramDataHandoff(), diagnostics);
  if (!programDataDigest)
    return programDataDigest.takeError();
  assembly->manifest.programData.digest = std::move(*programDataDigest);

  llvm::Expected<runtime::VerifiedPackageManifest> verified =
      runtime::verifyPackageManifest(std::move(assembly->manifest),
                                     stagingRoot);
  if (!verified)
    return fail(diagnostics, "package manifest verification failed: " +
                                 llvm::toString(verified.takeError()));
  if (llvm::Error error = writeManifest(stagingRoot, *verified))
    return fail(diagnostics, llvm::toString(std::move(error)));

  llvm::Expected<runtime::detail::BoundExecutablePackage> readback =
      runtime::detail::bindExecutablePackage(stagingRoot);
  if (!readback)
    return fail(diagnostics, "package manifest readback failed: " +
                                 llvm::toString(readback.takeError()));
  const runtime::PackageManifest &readbackManifest =
      readback->manifest.getManifest();
  const ExecutionConfig &executionConfig = cardExecutable.getExecutionConfig();
  const VerifiedTargetModule &targetReadback =
      targetModules.getModules().front();
  if (readbackManifest.cardCount != 1 ||
      readbackManifest.tileCount != executionConfig.getTileCount() ||
      readbackManifest.targetIdentity != targetReadback.getTargetIdentityId() ||
      readbackManifest.runtimeABI != targetReadback.getKernelRuntimeABIId() ||
      readbackManifest.launch != targetModules.getRuntimeLaunchContract() ||
      readbackManifest.moduleFormat != targetReadback.getModuleFormat() ||
      readbackManifest.targetIdentity != executionConfig.getTargetIdentityId())
    return fail(diagnostics,
                "package manifest readback does not match target module facts "
                "and ExecutionConfig");
  if (llvm::Error error = fsyncPackageTree(stagingRoot))
    return fail(diagnostics, llvm::toString(std::move(error)));
  if (renameDirectoryNoReplace(stagingRoot, outputDirectory, diagnostics))
    return llvm::createStringError(llvm::errc::io_error,
                                   "package directory rename failed");
  cleanup.release();
  return std::move(readback->manifest);
}

bool detail::doesPackageSlotMatchProgramBinding(
    const TileEntryArgument &slot, const ProgramResourceBinding &binding) {
  switch (slot.kind) {
  case TileEntryArgumentKind::ExternalInput:
    if (binding.role != ProgramResourceRole::UserInput)
      return false;
    break;
  case TileEntryArgumentKind::TargetTensor:
    if (binding.role != ProgramResourceRole::Parameter &&
        binding.role != ProgramResourceRole::Constant)
      return false;
    break;
  case TileEntryArgumentKind::ExternalOutput:
    if (binding.role != ProgramResourceRole::Output)
      return false;
    break;
  case TileEntryArgumentKind::Workspace:
  case TileEntryArgumentKind::ProfileRecord:
  case TileEntryArgumentKind::TransportStatus:
    return false;
  }
  return binding.index == slot.resourceIndex && binding.dtype == slot.dtype &&
         binding.localShape == slot.shape;
}

bool detail::isValidPackageCompilerManagedSlot(const TileEntryArgument &slot) {
  if (slot.layout != MemLayout::Tensor || slot.byteSize <= 0 ||
      slot.alignment <= 0)
    return false;
  if (slot.kind == TileEntryArgumentKind::Workspace)
    return slot.resourceIndex == 0 && slot.dtype == "u8" &&
           slot.shape.size() == 1 && slot.shape.front() == slot.byteSize;
  if (slot.kind == TileEntryArgumentKind::ProfileRecord)
    return slot.resourceIndex == 0 && slot.dtype == "u8" &&
           slot.shape.size() == 1 && slot.shape.front() == slot.byteSize &&
           slot.alignment == WAFER_TX81_PROFILER_BUFFER_ALIGNMENT &&
           (slot.byteSize == WAFER_TX81_PROFILER_MIN_BUFFER_BYTES ||
            slot.byteSize == WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES);
  if (slot.kind == TileEntryArgumentKind::TransportStatus)
    return slot.resourceIndex == 0 && slot.dtype == "u32" &&
           slot.shape == std::vector<int64_t>{1} &&
           slot.byteSize == runtime::kDirectDTEStatusStorageBytes &&
           slot.alignment == runtime::kDirectDTEStatusStorageAlignment;
  return false;
}

} // namespace wafer::compiler
