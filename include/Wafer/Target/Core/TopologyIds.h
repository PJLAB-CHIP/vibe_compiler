//===- TopologyIds.h - Typed target card and Tile identities -*- C++ -*-===//

#ifndef WAFER_TARGET_TOPOLOGYIDS_H
#define WAFER_TARGET_TOPOLOGYIDS_H

#include <cstdint>

namespace wafer {

/// Card identity in the target topology. This type is intentionally
/// distinct from GSPMD partition ordinals and runtime launch slots.
class CardId {
public:
  explicit constexpr CardId(int64_t value) : value(value) {}

  constexpr int64_t getValue() const { return value; }

  friend constexpr bool operator==(CardId lhs, CardId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(CardId lhs, CardId rhs) {
    return !(lhs == rhs);
  }

private:
  int64_t value;
};

/// Row-major Tile identity within one card. This type does not encode
/// a runtime launch slot and cannot be implicitly mixed with CardId.
class TileId {
public:
  explicit constexpr TileId(int64_t value) : value(value) {}

  constexpr int64_t getValue() const { return value; }

  friend constexpr bool operator==(TileId lhs, TileId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(TileId lhs, TileId rhs) {
    return !(lhs == rhs);
  }

private:
  int64_t value;
};

/// Dense runtime submission slot for one Tile entry. This is a
/// launch-table position, not a topology identity or GSPMD partition ordinal.
class LaunchSlotId {
public:
  explicit constexpr LaunchSlotId(int64_t value) : value(value) {}

  constexpr int64_t getValue() const { return value; }

  friend constexpr bool operator==(LaunchSlotId lhs, LaunchSlotId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(LaunchSlotId lhs, LaunchSlotId rhs) {
    return !(lhs == rhs);
  }

private:
  int64_t value;
};

} // namespace wafer

#endif // WAFER_TARGET_TOPOLOGYIDS_H
