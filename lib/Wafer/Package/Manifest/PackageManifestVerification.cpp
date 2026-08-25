//===- PackageManifestVerification.cpp - Manifest semantic verification --===//

#include "PackageManifestInternal.h"

#include "Wafer/ABI/Tx81ProfilerABI.h"
#include "Wafer/Package/Profile/ProfileInstrumentationModel.h"
#include "Wafer/Target/Layout/PhysicalTensorCodec.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace wafer::runtime {
namespace {

using detail::invalid;

bool isRegularFile(llvm::StringRef path) {
  return llvm::sys::fs::get_file_type(path, /*Follow=*/false) ==
         llvm::sys::fs::file_type::regular_file;
}

bool isPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

bool isValidAccess(PackageAccessMode access) {
  switch (access) {
  case PackageAccessMode::None:
  case PackageAccessMode::ReadOnly:
  case PackageAccessMode::WriteOnly:
  case PackageAccessMode::ReadWrite:
    return true;
  }
  return false;
}

bool isValidDigest(llvm::StringRef digest) {
  if (!digest.consume_front("sha256:") || digest.size() != 64)
    return false;
  return llvm::all_of(digest, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

bool isValidRelativePath(llvm::StringRef path) {
  if (path.empty() || llvm::sys::path::is_absolute(path) || path.contains('\\'))
    return false;
  llvm::SmallVector<llvm::StringRef, 8> components;
  path.split(components, '/', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  return llvm::all_of(components, [](llvm::StringRef component) {
    return !component.empty() && component != "." && component != "..";
  });
}

bool isValidShape(llvm::ArrayRef<int64_t> shape,
                  const PackageParseLimits &limits) {
  return shape.size() <= limits.maxShapeRank &&
         llvm::all_of(shape, [](int64_t dimension) { return dimension >= 0; });
}

llvm::Expected<PhysicalTensorDescriptor>
makePackageTensorKey(LogicalFormat dtype, PackageMemLayout layout,
                     llvm::ArrayRef<int64_t> shape, llvm::StringRef owner) {
  std::vector<uint64_t> unsignedShape;
  unsignedShape.reserve(shape.size());
  for (int64_t dimension : shape) {
    if (dimension < 0)
      return invalid(llvm::Twine(owner) + " has a negative tensor dimension");
    unsignedShape.push_back(static_cast<uint64_t>(dimension));
  }
  llvm::Expected<PhysicalTensorDescriptor> key =
      PhysicalTensorDescriptor::create(dtype, getPhysicalTensorLayout(layout),
                                       std::move(unsignedShape));
  if (!key)
    return invalid(llvm::Twine(owner) +
                   " is rejected by the physical tensor codec: " +
                   llvm::toString(key.takeError()));
  return key;
}

llvm::Expected<uint64_t>
verifyPhysicalTensorDescriptor(LogicalFormat dtype, PackageMemLayout layout,
                               llvm::ArrayRef<int64_t> shape, uint64_t bytes,
                               llvm::StringRef owner) {
  llvm::Expected<PhysicalTensorDescriptor> key =
      makePackageTensorKey(dtype, layout, shape, owner);
  if (!key)
    return key.takeError();
  llvm::Expected<uint64_t> expectedBytes = getPhysicalTensorStorageBytes(*key);
  if (!expectedBytes)
    return invalid(llvm::Twine(owner) + " physical geometry is invalid: " +
                   llvm::toString(expectedBytes.takeError()));
  if (*expectedBytes != bytes)
    return invalid(llvm::Twine(owner) +
                   " byte count disagrees with the physical tensor "
                   "codec (expected " +
                   llvm::Twine(*expectedBytes) + ", got " + llvm::Twine(bytes) +
                   ")");
  return key->getElementCount();
}

llvm::Expected<uint64_t> getLogicalElementCount(llvm::ArrayRef<int64_t> shape,
                                                llvm::StringRef owner) {
  uint64_t count = 1;
  for (int64_t dimension : shape) {
    if (dimension < 0 ||
        (dimension != 0 && count > std::numeric_limits<uint64_t>::max() /
                                       static_cast<uint64_t>(dimension)))
      return invalid(llvm::Twine(owner) +
                     " has an invalid or overflowing logical shape");
    count *= static_cast<uint64_t>(dimension);
  }
  return count;
}

llvm::Expected<std::string> digestFile(llvm::StringRef path) {
  if (!isRegularFile(path))
    return invalid("package member is not a regular file: " + path);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read package member: " + path);
  llvm::SHA256 hasher;
  hasher.update((*buffer)->getBuffer());
  return "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

template <typename Range, typename Projection>
bool hasDenseIds(const Range &records, Projection projection) {
  std::vector<bool> seen(records.size(), false);
  for (const auto &record : records) {
    uint64_t id = projection(record).getValue();
    if (id >= seen.size() || seen[id])
      return false;
    seen[id] = true;
  }
  return llvm::all_of(seen, [](bool value) { return value; });
}

PackageAccessMode
expectedArgumentAccess(const TileEntryArgumentReference &reference) {
  if (std::holds_alternative<ExternalInputArgument>(reference) ||
      std::holds_alternative<TargetTensorArgument>(reference))
    return PackageAccessMode::ReadOnly;
  if (std::holds_alternative<ExternalOutputArgument>(reference))
    return PackageAccessMode::WriteOnly;
  return PackageAccessMode::ReadWrite;
}

llvm::Error verifyProgramData(const PackageManifest &manifest,
                              llvm::StringRef packageRoot) {
  const ProgramDataRecord &programData = manifest.programData;
  const bool emptyTable = manifest.targetTensors.empty();
  if (programData.relativePath != kPackageProgramDataRelativePath)
    return invalid("package program_data relative path is not canonical");
  llvm::SmallString<256> path(packageRoot);
  llvm::sys::path::append(path, programData.relativePath);
  if (!isRegularFile(path))
    return invalid("package program data is not a regular file: " + path);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!buffer)
    return llvm::createStringError(
        buffer.getError(), "failed to read package program data: " + path);
  llvm::SHA256 hasher;
  hasher.update((*buffer)->getBuffer());
  const std::string digest =
      "sha256:" + llvm::toHex(hasher.final(), /*LowerCase=*/true);
  if (digest != programData.digest)
    return invalid("package program data digest mismatch");
  llvm::StringRef content = (*buffer)->getBuffer();
  if (emptyTable) {
    if (programData.totalBytes != 0 || programData.baseAlignment != 1)
      return invalid(
          "empty package program data must record zero bytes and unit "
          "alignment");
    if (digest != "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca4"
                  "95991b7852b855")
      return invalid("empty package program data must be the canonical empty "
                     "file");
    return llvm::Error::success();
  }
  if (programData.totalBytes == 0 || programData.baseAlignment == 0 ||
      !isPowerOfTwo(programData.baseAlignment))
    return invalid(
        "non-empty package program data needs positive bytes and a power-of-"
        "two base alignment");
  if (programData.totalBytes != content.size())
    return invalid("package program data byte count does not match the file");

  // Rebuild the canonical non-overlap placement with the same stable
  // tie-break the package writer uses, and prove the recorded ids, offsets,
  // padding and byte accounting follow it exactly.
  llvm::SmallVector<TargetTensorRecord, 4> canonical(
      manifest.targetTensors.begin(), manifest.targetTensors.end());
  llvm::sort(canonical,
             [](const TargetTensorRecord &lhs, const TargetTensorRecord &rhs) {
               return std::tie(lhs.programTensor, lhs.dtype, lhs.layout,
                               lhs.shape, lhs.bytes, lhs.alignment) <
                      std::tie(rhs.programTensor, rhs.dtype, rhs.layout,
                               rhs.shape, rhs.bytes, rhs.alignment);
             });
  uint64_t cursor = 0;
  uint64_t baseAlignment = 1;
  auto verifyZeroPadding = [&](uint64_t offset, uint64_t bytes) -> llvm::Error {
    constexpr uint64_t kWindowBytes = 1 << 20;
    uint64_t checked = 0;
    while (checked < bytes) {
      const uint64_t chunk = std::min<uint64_t>(kWindowBytes, bytes - checked);
      if (llvm::StringRef(content)
              .slice(offset + checked, offset + checked + chunk)
              .find_first_not_of('\0') != llvm::StringRef::npos)
        return invalid("package program data padding is not zero");
      checked += chunk;
    }
    return llvm::Error::success();
  };
  for (auto [index, tensor] : llvm::enumerate(canonical)) {
    if (tensor.id.getValue() != static_cast<uint64_t>(index))
      return invalid(
          "package target tensor ids do not follow the canonical order");
    if (cursor > std::numeric_limits<uint64_t>::max() - (tensor.alignment - 1))
      return invalid("package program data offset overflows");
    const uint64_t alignedCursor = llvm::alignTo(cursor, tensor.alignment);
    if (alignedCursor < cursor)
      return invalid("package target tensor ranges overlap");
    if (tensor.fileOffset != alignedCursor)
      return invalid("package target tensor offset is not canonical");
    if (alignedCursor > cursor) {
      if (llvm::Error error = verifyZeroPadding(cursor, alignedCursor - cursor))
        return error;
    }
    if (tensor.bytes > std::numeric_limits<uint64_t>::max() - alignedCursor)
      return invalid("package target tensor range overflows");
    cursor = alignedCursor + tensor.bytes;
    baseAlignment = std::max(baseAlignment, tensor.alignment);
  }
  if (programData.totalBytes < cursor)
    return invalid("package program data is shorter than the canonical "
                   "placement");
  if (programData.totalBytes > cursor)
    return invalid("package program data has trailing bytes after the "
                   "canonical placement");
  if (programData.baseAlignment != baseAlignment)
    return invalid("package program data base alignment is not canonical");
  return llvm::Error::success();
}

llvm::Error verifyRuntimeLaunchContract(const PackageManifest &manifest) {
  const KernelRuntimeLaunchContract &kernel = manifest.launch.getKernel();
  const bool kernelGrid = kernel.form == KernelLaunchForm::Grid;
  const bool cluster = kernel.form == KernelLaunchForm::Cluster;
  if (manifest.tileCount != 16 || manifest.entries.size() != 16)
    return invalid("kernel launches require a complete 16-Tile domain");

  std::vector<const PackageEntrypointRecord *> entriesByLaunchSlot(
      manifest.tileCount, nullptr);
  for (const PackageEntrypointRecord &entry : manifest.entries)
    entriesByLaunchSlot[entry.launchSlot.getValue()] = &entry;
  const PackageEntrypointRecord &first = *entriesByLaunchSlot.front();
  const PackageModuleRecord *firstModule =
      findModule(manifest.modules, first.module);
  if (!firstModule)
    return invalid("runtime launch entry references a missing module");

  llvm::DenseMap<uint64_t, uint64_t> moduleReferenceCount;
  for (const PackageEntrypointRecord &entry : manifest.entries)
    ++moduleReferenceCount[entry.module.getValue()];
  if (manifest.modules.size() != 1 ||
      llvm::any_of(manifest.entries, [&](const auto &entry) {
        return entry.module != first.module;
      }))
    return invalid(
        "grid/cluster launch requires one module referenced by all Tiles");

  auto exportRoleForPhase = [](RuntimeLaunchPhaseRole phase) {
    return phase == RuntimeLaunchPhaseRole::Prepare
               ? PackageModuleExportRole::Prepare
               : PackageModuleExportRole::Main;
  };
  for (const PackageModuleRecord &module : manifest.modules) {
    if (module.exports.size() != manifest.launch.getPhases().size())
      return invalid(
          "package module exports do not match the runtime launch phases");
    for (auto [index, phase] : llvm::enumerate(manifest.launch.getPhases()))
      if (module.exports[index].role != exportRoleForPhase(phase))
        return invalid(
            "package module export order does not match the runtime launch "
            "phases");
  }

  if (kernelGrid || cluster) {
    if (first.arguments.empty())
      return invalid("shared launch requires at least one typed argument");
    const uint64_t argumentBytesMax = cluster
                                          ? kTx81ClusterKernelArgumentBytesMax
                                          : kTx81KernelArgumentBytesMax;
    if (kernel.entryABI == KernelEntryABI::TileMajorPointerTable) {
      if (first.arguments.size() > argumentBytesMax / sizeof(uint64_t) / 16)
        return invalid(
            "shared Tile-major argument table exceeds the qualified V5.6 "
            "packet limit");
    } else if (kernel.entryABI == KernelEntryABI::TileRowPointerTable) {
      if (16 > argumentBytesMax / sizeof(uint64_t))
        return invalid(
            "shared Tile-row pointer table exceeds the qualified V5.6 "
            "packet limit");
    } else {
      return invalid("shared kernel launch has an incompatible entry ABI");
    }
  }

  const size_t firstTransportKind = first.transport.index();
  for (int64_t launchSlot = 0; launchSlot < manifest.tileCount; ++launchSlot) {
    const PackageEntrypointRecord &entry = *entriesByLaunchSlot[launchSlot];
    const PackageModuleRecord *module =
        findModule(manifest.modules, entry.module);
    const PackageModuleExportRecord *main =
        module ? findModuleExport(*module, PackageModuleExportRole::Main)
               : nullptr;
    if (!module || !main)
      return invalid("runtime launch entry has no typed main export");
    if (entry.transport.index() != firstTransportKind)
      return invalid("runtime launch Tiles have mixed transport contracts");
    if (entry.arguments.size() != first.arguments.size())
      return invalid("multi-tile launch entries have different argument "
                     "counts");
    for (size_t index = 0; index < entry.arguments.size(); ++index)
      if (entry.arguments[index].reference.index() !=
          first.arguments[index].reference.index())
        return invalid("multi-Tile launch argument kinds are inconsistent");
    if (const auto *requirements =
            std::get_if<DirectDTETransportRequirements>(&entry.transport)) {
      const auto *firstRequirements =
          std::get_if<DirectDTETransportRequirements>(&first.transport);
      if (!firstRequirements ||
          requirements->statusABI != firstRequirements->statusABI ||
          requirements->hostWatchdogRequired !=
              firstRequirements->hostWatchdogRequired)
        return invalid(
            "Direct DTE Tiles have inconsistent transport contracts");
    }
  }
  return llvm::Error::success();
}

llvm::Error verifyModuleFiles(const PackageManifest &manifest,
                              llvm::StringRef packageRoot) {
  llvm::SmallString<256> modulesRoot(packageRoot);
  llvm::sys::path::append(modulesRoot, "modules");
  if (llvm::sys::fs::get_file_type(modulesRoot, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("package modules directory is missing or not a directory");

  llvm::StringSet<> expectedPaths;
  llvm::StringSet<> expectedDirectories;
  expectedDirectories.insert(modulesRoot);
  for (const PackageModuleRecord &module : manifest.modules) {
    llvm::SmallString<256> path(packageRoot);
    llvm::sys::path::append(path, module.relativePath);
    llvm::Expected<std::string> digest = digestFile(path);
    if (!digest)
      return digest.takeError();
    if (*digest != module.digest)
      return invalid("package module digest mismatch: " + module.relativePath);
    expectedPaths.insert(path);
    llvm::SmallString<256> parent(path);
    llvm::sys::path::remove_filename(parent);
    while (parent != modulesRoot) {
      if (parent.size() <= modulesRoot.size())
        return invalid("package module path escapes the modules directory");
      expectedDirectories.insert(parent);
      llvm::sys::path::remove_filename(parent);
    }
  }

  std::error_code error;
  for (llvm::sys::fs::recursive_directory_iterator
           iterator(modulesRoot, error, /*follow_symlinks=*/false),
       end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return invalid("failed to walk package modules: " + error.message());
    llvm::sys::fs::file_type type = iterator->type();
    if (type == llvm::sys::fs::file_type::directory_file) {
      if (!expectedDirectories.contains(iterator->path()))
        return invalid("package modules contain an undeclared directory: " +
                       iterator->path());
      continue;
    }
    if (type != llvm::sys::fs::file_type::regular_file)
      return invalid("package modules contain a non-regular member");
    if (!expectedPaths.contains(iterator->path()))
      return invalid("package modules contain an unreferenced payload: " +
                     iterator->path());
  }
  if (error)
    return invalid("failed to walk package modules: " + error.message());
  return llvm::Error::success();
}

} // namespace

namespace detail {

llvm::Error verifyPackageRoot(const PackageManifest &manifest,
                              llvm::StringRef packageRoot) {
  // The package root is an all-and-only closure: exactly the canonical
  // manifest, the modules directory and the data directory with exactly
  // program-data.bin. Extra regular files, directories, symlinks and any
  // unreferenced payload are rejected.
  constexpr llvm::StringLiteral kManifestName = "manifest.json";
  constexpr llvm::StringLiteral kModulesDirectory = "modules";
  constexpr llvm::StringLiteral kDataDirectory = "data";
  constexpr llvm::StringLiteral kProgramDataName = "program-data.bin";
  std::set<std::string> allowedRootMembers = {
      (packageRoot + llvm::sys::path::get_separator() + kManifestName).str()};
  allowedRootMembers.insert(
      (packageRoot + llvm::sys::path::get_separator() + kModulesDirectory)
          .str());
  allowedRootMembers.insert(
      (packageRoot + llvm::sys::path::get_separator() + kDataDirectory).str());

  llvm::SmallString<256> manifestPath(packageRoot);
  llvm::sys::path::append(manifestPath, kManifestName);
  if (llvm::sys::fs::get_file_type(manifestPath, /*Follow=*/false) !=
      llvm::sys::fs::file_type::regular_file)
    return invalid("package root is missing its canonical manifest file");
  llvm::SmallString<256> modulesPath(packageRoot);
  llvm::sys::path::append(modulesPath, kModulesDirectory);
  if (llvm::sys::fs::get_file_type(modulesPath, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("package root is missing its modules directory");
  llvm::SmallString<256> dataPath(packageRoot);
  llvm::sys::path::append(dataPath, kDataDirectory);
  if (llvm::sys::fs::get_file_type(dataPath, /*Follow=*/false) !=
      llvm::sys::fs::file_type::directory_file)
    return invalid("package root is missing its data directory");

  std::error_code error;
  for (llvm::sys::fs::directory_iterator iterator(packageRoot, error), end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return invalid("failed to walk package root: " + error.message());
    if (iterator->type() != llvm::sys::fs::file_type::regular_file &&
        iterator->type() != llvm::sys::fs::file_type::directory_file)
      return invalid("package root contains an unsupported member: " +
                     iterator->path());
    if (allowedRootMembers.count(iterator->path()) == 0)
      return invalid("package root contains an undeclared member: " +
                     iterator->path());
  }
  if (error)
    return invalid("failed to walk package root: " + error.message());

  // data/ holds exactly the canonical program-data member.
  llvm::SmallString<256> programDataPath(dataPath);
  llvm::sys::path::append(programDataPath, kProgramDataName);
  if (llvm::sys::fs::get_file_type(programDataPath, /*Follow=*/false) !=
      llvm::sys::fs::file_type::regular_file)
    return invalid("package data directory is missing program-data.bin");
  for (llvm::sys::fs::directory_iterator iterator(dataPath, error), end;
       iterator != end; iterator.increment(error)) {
    if (error)
      return invalid("failed to walk package data directory: " +
                     error.message());
    if (iterator->type() != llvm::sys::fs::file_type::regular_file)
      return invalid("package data directory contains an unsupported "
                     "member: " +
                     iterator->path());
    if (iterator->path() != programDataPath.str())
      return invalid("package data directory contains an undeclared "
                     "member: " +
                     iterator->path());
  }
  if (error)
    return invalid("failed to walk package data directory: " + error.message());
  return llvm::Error::success();
}

} // namespace detail

llvm::Expected<VerifiedPackageManifest>
verifyPackageManifest(PackageManifest manifest, llvm::StringRef packageRoot,
                      const PackageParseLimits &limits) {
  if (!manifest.program.isValid() || manifest.program.getValue() != 0)
    return invalid("package program identity is invalid");
  if (manifest.targetIdentity != TargetIdentityId::waferTx81SingleCard() ||
      manifest.runtimeABI != KernelRuntimeABIId::waferTx81Kernel() ||
      manifest.moduleFormat != kCurrentTargetModuleFormat)
    return invalid("package target identity or runtime ABI is unsupported");
  if (manifest.cardCount != 1)
    return invalid("package card_count must be exactly 1");
  if (manifest.tileCount != 16)
    return invalid("package tile_count must be exactly 16");
  uint64_t totalRecords = manifest.programTensors.size() +
                          manifest.targetTensors.size() +
                          manifest.inputs.size() + manifest.outputs.size() +
                          manifest.modules.size() + manifest.entries.size();
  if (totalRecords > limits.maxRecords)
    return invalid("package manifest exceeds record limit");
  for (const PackageEntrypointRecord &entry : manifest.entries) {
    if (entry.arguments.size() > limits.maxRecords - totalRecords)
      return invalid("package manifest exceeds record limit");
    totalRecords += entry.arguments.size();
  }
  for (const PackageModuleRecord &module : manifest.modules) {
    if (module.exports.size() > limits.maxRecords - totalRecords)
      return invalid("package manifest exceeds record limit");
    totalRecords += module.exports.size();
  }
  if (manifest.modules.empty() ||
      manifest.entries.size() != static_cast<uint64_t>(manifest.tileCount))
    return invalid("package module/entry Tile domain is incomplete");
  if (!hasDenseIds(manifest.programTensors,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.targetTensors,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.inputs,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.outputs,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.modules,
                   [](const auto &record) { return record.id; }) ||
      !hasDenseIds(manifest.entries,
                   [](const auto &record) { return record.id; }))
    return invalid("package IDs must be unique dense zero-based domains");

  llvm::sort(manifest.programTensors,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.targetTensors,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.inputs,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.outputs,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.modules,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });
  llvm::sort(manifest.entries,
             [](const auto &lhs, const auto &rhs) { return lhs.id < rhs.id; });

  std::set<std::pair<ProgramTensorRole, int64_t>> programTensorRoles;
  std::vector<uint64_t> programTensorElementCounts(
      manifest.programTensors.size(), 0);
  for (const ProgramTensorRecord &tensor : manifest.programTensors) {
    if (tensor.roleIndex < 0 ||
        tensor.roleIndex > std::numeric_limits<uint32_t>::max() ||
        !isValidShape(tensor.globalShape, limits) ||
        !isValidShape(tensor.localShape, limits) ||
        !isValidShape(tensor.sliceOffsets, limits) ||
        !isValidShape(tensor.sliceSizes, limits) ||
        tensor.sliceOffsets.size() != tensor.globalShape.size() ||
        tensor.sliceSizes.size() != tensor.globalShape.size() ||
        tensor.localShape != tensor.sliceSizes)
      return invalid("package program tensor has invalid identity or shape");
    if (!programTensorRoles.insert({tensor.role, tensor.roleIndex}).second)
      return invalid("package program tensor role/index is duplicated");
    llvm::Expected<uint64_t> localKey = getLogicalElementCount(
        tensor.localShape, "package program tensor local descriptor");
    if (!localKey)
      return localKey.takeError();
    llvm::Expected<uint64_t> globalKey = getLogicalElementCount(
        tensor.globalShape, "package program tensor global descriptor");
    if (!globalKey)
      return globalKey.takeError();
    programTensorElementCounts[tensor.id.getValue()] = *localKey;
  }

  std::set<std::tuple<ProgramTensorId, LogicalFormat, PackageMemLayout,
                      std::vector<int64_t>, uint64_t, uint64_t>>
      targetTensorDescriptors;
  for (const TargetTensorRecord &tensor : manifest.targetTensors) {
    if (!findProgramTensor(manifest.programTensors, tensor.programTensor))
      return invalid("package target tensor references a missing program "
                     "tensor");
    if (!isValidShape(tensor.shape, limits) || tensor.bytes == 0 ||
        tensor.bytes >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        tensor.alignment == 0 || !isPowerOfTwo(tensor.alignment) ||
        tensor.fileOffset % tensor.alignment != 0)
      return invalid("package target tensor has invalid descriptor or offset");
    if (!targetTensorDescriptors
             .insert({tensor.programTensor, tensor.dtype, tensor.layout,
                      tensor.shape, tensor.bytes, tensor.alignment})
             .second)
      return invalid("package target tensor descriptors are duplicated");
    llvm::Expected<uint64_t> elementCount = verifyPhysicalTensorDescriptor(
        tensor.dtype, tensor.layout, tensor.shape, tensor.bytes,
        "package target tensor descriptor");
    if (!elementCount)
      return elementCount.takeError();
    if (*elementCount !=
        programTensorElementCounts[tensor.programTensor.getValue()])
      return invalid("package target tensor element count disagrees with its "
                     "program tensor");
    if (tensor.fileOffset >
            std::numeric_limits<uint64_t>::max() - tensor.bytes ||
        tensor.fileOffset + tensor.bytes > manifest.programData.totalBytes)
      return invalid("package target tensor range is outside program data");
  }

  llvm::DenseSet<int64_t> inputRoleIndices;
  for (const ExternalPortRecord &port : manifest.inputs) {
    if (port.roleIndex < 0 ||
        port.roleIndex > std::numeric_limits<uint32_t>::max() ||
        !isValidShape(port.logicalShape, limits) ||
        !isValidShape(port.shape, limits) || port.bytes == 0 ||
        port.bytes >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        port.alignment == 0 || !isPowerOfTwo(port.alignment))
      return invalid("package input port has invalid descriptor");
    llvm::Expected<uint64_t> logicalKey = getLogicalElementCount(
        port.logicalShape, "package input logical descriptor");
    if (!logicalKey)
      return logicalKey.takeError();
    llvm::Expected<uint64_t> physicalElementCount =
        verifyPhysicalTensorDescriptor(port.dtype, port.layout, port.shape,
                                       port.bytes,
                                       "package input target descriptor");
    if (!physicalElementCount)
      return physicalElementCount.takeError();
    if (*physicalElementCount != *logicalKey)
      return invalid("package input logical and target element counts "
                     "disagree");
    if (!inputRoleIndices.insert(port.roleIndex).second)
      return invalid("package input port role index is duplicated");
  }
  llvm::DenseSet<int64_t> outputRoleIndices;
  for (const ExternalPortRecord &port : manifest.outputs) {
    if (port.roleIndex < 0 ||
        port.roleIndex > std::numeric_limits<uint32_t>::max() ||
        !isValidShape(port.logicalShape, limits) ||
        !isValidShape(port.shape, limits) || port.bytes == 0 ||
        port.bytes >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        port.alignment == 0 || !isPowerOfTwo(port.alignment))
      return invalid("package output port has invalid descriptor");
    llvm::Expected<uint64_t> logicalKey = getLogicalElementCount(
        port.logicalShape, "package output logical descriptor");
    if (!logicalKey)
      return logicalKey.takeError();
    llvm::Expected<uint64_t> physicalElementCount =
        verifyPhysicalTensorDescriptor(port.dtype, port.layout, port.shape,
                                       port.bytes,
                                       "package output target descriptor");
    if (!physicalElementCount)
      return physicalElementCount.takeError();
    if (*physicalElementCount != *logicalKey)
      return invalid("package output logical and target element counts "
                     "disagree");
    if (!outputRoleIndices.insert(port.roleIndex).second)
      return invalid("package output port role index is duplicated");
  }

  llvm::StringSet<> modulePaths;
  for (const PackageModuleRecord &module : manifest.modules) {
    if (!isValidRelativePath(module.relativePath) ||
        module.relativePath.size() > limits.maxStringBytes ||
        !llvm::StringRef(module.relativePath).starts_with("modules/") ||
        !modulePaths.insert(module.relativePath).second ||
        module.digest.size() > limits.maxStringBytes ||
        !isValidDigest(module.digest) ||
        module.format.size() > limits.maxStringBytes ||
        module.format != manifest.moduleFormat)
      return invalid("package module path/digest/format is invalid");
    if (module.exports.empty())
      return invalid("package module export set is empty");
    bool seenPrepare = false;
    bool seenMain = false;
    llvm::StringSet<> symbols;
    for (const PackageModuleExportRecord &moduleExport : module.exports) {
      bool *seen = nullptr;
      switch (moduleExport.role) {
      case PackageModuleExportRole::Prepare:
        seen = &seenPrepare;
        break;
      case PackageModuleExportRole::Main:
        seen = &seenMain;
        break;
      }
      if (*seen || moduleExport.symbol.empty() ||
          moduleExport.symbol.size() > limits.maxStringBytes ||
          llvm::StringRef(moduleExport.symbol).contains('\0') ||
          !symbols.insert(moduleExport.symbol).second)
        return invalid("package module export role/symbol is invalid");
      *seen = true;
    }
  }

  std::vector<uint64_t> targetTensorReferenceCounts(
      manifest.targetTensors.size(), 0);
  struct CardWorkspaceUse {
    uint64_t bytes = 0;
    uint64_t alignment = 0;
    bool read = false;
    bool write = false;
  };
  std::map<uint64_t, CardWorkspaceUse> cardWorkspaces;
  std::vector<uint64_t> inputReferenceCounts(manifest.inputs.size(), 0);
  std::vector<uint64_t> outputReferenceCounts(manifest.outputs.size(), 0);

  std::vector<bool> seenLaunchSlots(manifest.tileCount, false);
  llvm::DenseSet<int64_t> seenTileIds;
  llvm::DenseMap<uint64_t, uint64_t> referencedModules;
  for (PackageEntrypointRecord &entry : manifest.entries) {
    if (entry.cardId != CardId(0) || entry.tileId.getValue() < 0 ||
        entry.tileId.getValue() >= 16 || !entry.launchSlot.isValid() ||
        entry.launchSlot.getValue() >=
            static_cast<uint64_t>(manifest.tileCount) ||
        seenLaunchSlots[entry.launchSlot.getValue()] ||
        !seenTileIds.insert(entry.tileId.getValue()).second ||
        entry.completion != PackageEntryCompletionKind::ReturnAfterLocalDrain)
      return invalid("package entry Tile domain is invalid");
    seenLaunchSlots[entry.launchSlot.getValue()] = true;
    const PackageModuleRecord *module =
        findModule(manifest.modules, entry.module);
    if (!module)
      return invalid("package entry module relation is invalid");
    ++referencedModules[entry.module.getValue()];
    llvm::sort(entry.arguments, [](const auto &lhs, const auto &rhs) {
      return lhs.ordinal < rhs.ordinal;
    });
    for (auto [ordinal, argument] : llvm::enumerate(entry.arguments)) {
      const bool cardWorkspace =
          std::holds_alternative<CardWorkspaceArgument>(argument.reference);
      if (!isValidAccess(argument.access) || argument.ordinal != ordinal ||
          (!cardWorkspace &&
           argument.access != expectedArgumentAccess(argument.reference)))
        return invalid("package entry arguments must be dense, zero-based "
                       "and access-consistent");
      if (const auto *reference =
              std::get_if<ExternalInputArgument>(&argument.reference)) {
        if (!reference->port.isValid() ||
            reference->port.getValue() >= inputReferenceCounts.size())
          return invalid("package entry references a missing input port");
        ++inputReferenceCounts[reference->port.getValue()];
      } else if (const auto *reference =
                     std::get_if<TargetTensorArgument>(&argument.reference)) {
        if (!reference->tensor.isValid() ||
            reference->tensor.getValue() >= targetTensorReferenceCounts.size())
          return invalid("package entry references a missing TargetTensor");
        ++targetTensorReferenceCounts[reference->tensor.getValue()];
      } else if (const auto *reference =
                     std::get_if<ExternalOutputArgument>(&argument.reference)) {
        if (!reference->port.isValid() ||
            reference->port.getValue() >= outputReferenceCounts.size())
          return invalid("package entry references a missing output port");
        ++outputReferenceCounts[reference->port.getValue()];
      } else if (std::holds_alternative<WorkspaceArgument>(
                     argument.reference)) {
        const auto &workspace = std::get<WorkspaceArgument>(argument.reference);
        if (workspace.bytes == 0 || workspace.alignment == 0 ||
            !isPowerOfTwo(workspace.alignment))
          return invalid("package entry workspace requirement is invalid");
      } else if (const auto *workspace =
                     std::get_if<CardWorkspaceArgument>(&argument.reference)) {
        if (workspace->resource == std::numeric_limits<uint64_t>::max() ||
            workspace->bytes == 0 || workspace->alignment == 0 ||
            !isPowerOfTwo(workspace->alignment))
          return invalid("package entry card workspace requirement is invalid");
        auto [use, inserted] = cardWorkspaces.try_emplace(
            workspace->resource,
            CardWorkspaceUse{workspace->bytes, workspace->alignment});
        if (!inserted && (use->second.bytes != workspace->bytes ||
                          use->second.alignment != workspace->alignment))
          return invalid(
              "package card workspace references disagree on storage");
        use->second.read |= argument.access == PackageAccessMode::ReadOnly ||
                            argument.access == PackageAccessMode::ReadWrite;
        use->second.write |= argument.access == PackageAccessMode::WriteOnly ||
                             argument.access == PackageAccessMode::ReadWrite;
      } else if (std::holds_alternative<ProfileRecordArgument>(
                     argument.reference)) {
        const auto &profile =
            std::get<ProfileRecordArgument>(argument.reference);
        if (profile.recordABI != kProfileRecordABI || profile.bytes == 0 ||
            profile.alignment == 0 || !isPowerOfTwo(profile.alignment) ||
            profile.alignment != WAFER_TX81_PROFILER_BUFFER_ALIGNMENT ||
            (profile.bytes != WAFER_TX81_PROFILER_MIN_BUFFER_BYTES &&
             profile.bytes != WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES))
          return invalid("package entry profile record requirement is "
                         "invalid");
      } else {
        const auto &status =
            std::get<TransportStatusArgument>(argument.reference);
        if (status.statusABI != kDirectDTEStatusABI ||
            status.bytes != kDirectDTEStatusStorageBytes ||
            status.alignment != kDirectDTEStatusStorageAlignment)
          return invalid("package entry Direct DTE status requirement is "
                         "invalid");
      }
    }
    if (std::holds_alternative<NoTransportRequirements>(entry.transport)) {
      if (llvm::any_of(entry.arguments, [](const auto &argument) {
            return std::holds_alternative<TransportStatusArgument>(
                argument.reference);
          }))
        return invalid(
            "package transport status exists without Direct DTE requirement");
    } else {
      const auto &requirements =
          std::get<DirectDTETransportRequirements>(entry.transport);
      const auto statusCount =
          llvm::count_if(entry.arguments, [](const auto &argument) {
            return std::holds_alternative<TransportStatusArgument>(
                argument.reference);
          });
      if (statusCount != 1 || requirements.statusABI != kDirectDTEStatusABI ||
          !requirements.hostWatchdogRequired)
        return invalid("package Direct DTE transport requirement is invalid");
    }
  }
  uint64_t expectedCardWorkspace = 0;
  for (const auto &[resource, use] : cardWorkspaces)
    if (resource != expectedCardWorkspace++ || !use.read || !use.write)
      return invalid("package card workspaces must be dense and have both "
                     "reader and writer entries");
  if (llvm::any_of(targetTensorReferenceCounts,
                   [](uint64_t count) { return count == 0; }))
    return invalid(
        "package TargetTensors are not covered all-and-only by arguments");
  for (auto [port, count] : llvm::zip(manifest.inputs, inputReferenceCounts))
    if (count != static_cast<uint64_t>(manifest.tileCount))
      return invalid("package input port is not referenced by every Tile");
  for (auto [port, count] : llvm::zip(manifest.outputs, outputReferenceCounts))
    if (count != static_cast<uint64_t>(manifest.tileCount))
      return invalid("package output port is not referenced by every Tile");
  if (llvm::any_of(manifest.modules, [&](const auto &module) {
        return referencedModules.lookup(module.id.getValue()) == 0;
      }))
    return invalid("package contains an unreferenced module");

  if (llvm::Error error = verifyRuntimeLaunchContract(manifest))
    return std::move(error);

  if (packageRoot.empty())
    return invalid("package root must not be empty");
  if (llvm::Error error = verifyModuleFiles(manifest, packageRoot))
    return std::move(error);
  if (llvm::Error error = verifyProgramData(manifest, packageRoot))
    return std::move(error);
  return VerifiedPackageManifest(std::move(manifest));
}

} // namespace wafer::runtime
