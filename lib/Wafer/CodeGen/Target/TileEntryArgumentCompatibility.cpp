//===- TileEntryArgumentCompatibility.cpp - Multi-Tile entry order ------===//

#include "Wafer/CodeGen/Target/TargetCodeGenInternal.h"

#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {

std::optional<TileEntryArgumentOrderDifference>
findTileEntryArgumentOrderDifference(
    llvm::ArrayRef<TileEntryArgument> lhs,
    llvm::ArrayRef<TileEntryArgument> rhs) {
  if (lhs.size() != rhs.size())
    return TileEntryArgumentOrderDifference{
        std::min(lhs.size(), rhs.size()), "slot-count"};
  for (auto [index, slots] : llvm::enumerate(llvm::zip(lhs, rhs))) {
    const TileEntryArgument &left = std::get<0>(slots);
    const TileEntryArgument &right = std::get<1>(slots);
    if (left.ordinal != right.ordinal)
      return TileEntryArgumentOrderDifference{index, "ordinal"};
    if (left.kind != right.kind)
      return TileEntryArgumentOrderDifference{index, "kind"};
    if (left.resourceIndex != right.resourceIndex)
      return TileEntryArgumentOrderDifference{index, "resource-index"};
    if (left.access != right.access)
      return TileEntryArgumentOrderDifference{index, "access"};

    // Workspace capacity is a Tile-local resource fact. Runtime manifests
    // retain it per entry and invocation planning allocates it per launch
    // slot; only its typed position in the pointer row is shared.
    if (left.kind == TileEntryArgumentKind::Workspace)
      continue;

    if (left.dtype != right.dtype)
      return TileEntryArgumentOrderDifference{index, "dtype"};
    if (left.layout != right.layout)
      return TileEntryArgumentOrderDifference{index, "layout"};
    if (left.shape != right.shape)
      return TileEntryArgumentOrderDifference{index, "shape"};
    if (left.byteSize != right.byteSize)
      return TileEntryArgumentOrderDifference{index, "byte-size"};
    if (left.alignment != right.alignment)
      return TileEntryArgumentOrderDifference{index, "alignment"};
  }
  return std::nullopt;
}

} // namespace wafer::compiler::detail
