//===- TransportContract.h - Target transport selection -------*- C++ -*-===//

#ifndef WAFER_TARGET_TRANSPORTCONTRACT_H
#define WAFER_TARGET_TRANSPORTCONTRACT_H

#include <cstdint>

namespace wafer {

/// Closed transport contract carried by one accepted Tile executable.
enum class TransportContract : uint8_t { None, DirectDTE };

} // namespace wafer

#endif // WAFER_TARGET_TRANSPORTCONTRACT_H
