//===- PhysicalIds.h - Typed physical card and Tile identities -*- C++ -*-===//

#ifndef WAFER_TARGET_PHYSICALIDS_H
#define WAFER_TARGET_PHYSICALIDS_H

#include <cstdint>

namespace wafer {

/// Physical-card identity in the target topology. This type is intentionally
/// distinct from GSPMD partition ordinals and runtime launch slots.
class PhysicalCardId {
public:
  explicit constexpr PhysicalCardId(int64_t value) : value(value) {}

  constexpr int64_t getValue() const { return value; }

  friend constexpr bool operator==(PhysicalCardId lhs, PhysicalCardId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(PhysicalCardId lhs, PhysicalCardId rhs) {
    return !(lhs == rhs);
  }

private:
  int64_t value;
};

/// Row-major physical Tile identity within one card. This type does not encode
/// a runtime launch slot and cannot be implicitly mixed with PhysicalCardId.
class PhysicalTileId {
public:
  explicit constexpr PhysicalTileId(int64_t value) : value(value) {}

  constexpr int64_t getValue() const { return value; }

  friend constexpr bool operator==(PhysicalTileId lhs, PhysicalTileId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(PhysicalTileId lhs, PhysicalTileId rhs) {
    return !(lhs == rhs);
  }

private:
  int64_t value;
};

/// Dense runtime submission slot for one physical Tile entry. This is a
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

#endif // WAFER_TARGET_PHYSICALIDS_H
