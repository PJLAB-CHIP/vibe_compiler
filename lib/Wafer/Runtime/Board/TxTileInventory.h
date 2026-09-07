//===- TxTileInventory.h - TX physical Tile inventory adapter --*- C++ -*-===//

#ifndef WAFER_RUNTIME_BOARD_TXTILEINVENTORY_H
#define WAFER_RUNTIME_BOARD_TXTILEINVENTORY_H

#include "Wafer/Runtime/Board/BoardRuntime.h"

#include <optional>

namespace wafer::runtime {

inline std::optional<BoardDeviceInfo::Tile>
decodeTxTileInventory(uint16_t index, bool available, uint32_t physicalX,
                      uint32_t physicalY) {
  if (physicalX >= 4 || physicalY >= 4)
    return std::nullopt;
  // TX full-card logical IDs are X-major. The compiler topology's row/column
  // correspond to SDK physical X/Y respectively; these coordinate names are
  // not the compiler's Cartesian x/y. Keep the SDK submission index separate.
  return BoardDeviceInfo::Tile{TileId(physicalX * 4 + physicalY),
                               LaunchSlotId(index), available, physicalX,
                               physicalY};
}

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_BOARD_TXTILEINVENTORY_H
