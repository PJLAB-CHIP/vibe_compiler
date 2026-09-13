//===- TileEntryArgumentCompatibility.cpp - Tile resource agreement -----===//

#include "Wafer/CodeGen/LLVM/TargetCodeGenInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"

#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

llvm::StringRef findDifference(const TileEntryArgument &left,
                               const TileEntryArgument &right) {
  if (left.kind != right.kind)
    return "kind";
  if (left.resourceIndex != right.resourceIndex)
    return "resource-index";
  if (left.zeroInitialize != right.zeroInitialize)
    return "zero-initialize";
  if (left.kind != TileEntryArgumentKind::SharedWorkspace &&
      left.access != right.access)
    return "access";
  if (left.targetTensorMaterialization != right.targetTensorMaterialization)
    return "target-tensor-materialization";
  // Workspace storage is entry-local. Its size need not agree across Tiles.
  if (left.kind == TileEntryArgumentKind::Workspace)
    return {};
  if (left.dtype != right.dtype)
    return "dtype";
  if (left.layout != right.layout)
    return "layout";
  if (left.shape != right.shape)
    return "shape";
  if (left.byteSize != right.byteSize)
    return "byte-size";
  if (left.alignment != right.alignment)
    return "alignment";
  return {};
}

} // namespace

llvm::Error
validateTileEntryArgumentDomain(llvm::ArrayRef<TargetLLVMModule> modules) {
  std::map<int64_t, const TileEntryArgument *> sharedResources;
  llvm::SmallVector<const TileEntryArgument *> commonArguments;
  bool first = true;
  for (const auto &module : modules) {
    std::set<int64_t> localSharedResources;
    size_t commonIndex = 0;
    for (auto [index, slot] : llvm::enumerate(module.getTileEntryArguments())) {
      if (slot.ordinal != static_cast<int64_t>(index))
        return llvm::createStringError(llvm::errc::invalid_argument,
                                       "Tile entry arguments are not dense");
      llvm::StringRef difference;
      if (slot.kind == TileEntryArgumentKind::SharedWorkspace) {
        if (slot.resourceIndex < 0 ||
            !localSharedResources.insert(slot.resourceIndex).second)
          return llvm::createStringError(llvm::errc::invalid_argument,
                                         "Tile has an invalid or duplicate "
                                         "shared workspace resource");
        auto [found, inserted] =
            sharedResources.try_emplace(slot.resourceIndex, &slot);
        if (!inserted)
          difference = findDifference(slot, *found->second);
      } else {
        if (first)
          commonArguments.push_back(&slot);
        else if (commonIndex >= commonArguments.size())
          difference = "slot-count";
        else
          difference = findDifference(slot, *commonArguments[commonIndex]);
        ++commonIndex;
      }
      if (!difference.empty())
        return llvm::createStringError(
            llvm::errc::invalid_argument,
            "multi-Tile runtime launch requires compatible per-Tile resource "
            "bindings; launch_slot=%lld slot=%zu field=%s",
            static_cast<long long>(module.getLaunchSlotId().getValue()), index,
            difference.str().c_str());
    }
    if (commonIndex != commonArguments.size())
      return llvm::createStringError(llvm::errc::invalid_argument,
                                     "multi-Tile program argument domain "
                                     "is inconsistent");
    first = false;
  }
  return llvm::Error::success();
}

} // namespace wafer::compiler::detail
