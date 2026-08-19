//===- StaticBufferRange.h - Proven static buffer intervals -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_STATICBUFFERRANGE_H
#define WAFER_ANALYSIS_STATICBUFFERRANGE_H

#include "mlir/IR/Value.h"

#include <cstdint>
#include <optional>

namespace wafer::analysis {

/// A non-negative half-open byte interval. Instances returned by this analysis
/// always satisfy `begin <= end`; callers must not construct an invalid range
/// and use it as a legality proof.
struct StaticByteRange {
  int64_t begin = 0;
  int64_t end = 0;

  int64_t length() const { return end - begin; }

  friend bool operator==(const StaticByteRange &lhs,
                         const StaticByteRange &rhs) {
    return lhs.begin == rhs.begin && lhs.end == rhs.end;
  }
  friend bool operator!=(const StaticByteRange &lhs,
                         const StaticByteRange &rhs) {
    return !(lhs == rhs);
  }
};

/// One contiguous interval normalized to an allocation or public function
/// boundary root. This is a recomputable analysis value, never persisted as an
/// IR side channel.
struct StaticBufferRange {
  mlir::Value root;
  StaticByteRange bytes;

  friend bool operator==(const StaticBufferRange &lhs,
                         const StaticBufferRange &rhs) {
    return lhs.root == rhs.root && lhs.bytes == rhs.bytes;
  }
  friend bool operator!=(const StaticBufferRange &lhs,
                         const StaticBufferRange &rhs) {
    return !(lhs == rhs);
  }
};

/// Return the absolute byte interval encoded by a static Wafer memref type.
/// Dynamic, overflowing, bit-packed, blocked, overlapping, and non-contiguous
/// layouts fail closed.
std::optional<StaticByteRange> getStaticByteRange(mlir::Value value);

/// Resolve static rank-preserving subviews, compatible memref casts, and
/// tile-region boundary aliases to one allocation/public-boundary root. Every
/// subview offset and stride is recomputed from its source layout before the
/// leaf interval is normalized to the root.
std::optional<StaticBufferRange> resolveStaticBufferRange(mlir::Value value);

bool staticByteRangesOverlap(const StaticByteRange &lhs,
                             const StaticByteRange &rhs);
bool staticByteRangesAreDisjoint(const StaticByteRange &lhs,
                                 const StaticByteRange &rhs);
bool staticByteRangeContains(const StaticByteRange &container,
                             const StaticByteRange &contained);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_STATICBUFFERRANGE_H
