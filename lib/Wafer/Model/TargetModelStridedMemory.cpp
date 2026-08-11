//===- TargetModelStridedMemory.cpp - Strided target model memory --------===//

#include "Wafer/Model/TargetModelMemory.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <limits>
#include <tuple>

namespace wafer::model {
namespace {

llvm::Error memoryError(TargetModelMemoryErrorCode code,
                        const llvm::Twine &detail) {
  return llvm::make_error<TargetModelMemoryError>(code, detail.str());
}

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool isPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

struct StridedLayoutInfo {
  uint64_t segmentCount = 0;
  uint64_t payloadBytes = 0;
  uint64_t boundingSpan = 0;
  bool analyticallyNonOverlapping = false;
};

llvm::Expected<StridedLayoutInfo>
inspectStridedLayout(const TargetModelStridedByteLayout &layout) {
  if (layout.innerBytes == 0)
    return memoryError(TargetModelMemoryErrorCode::InvalidEffect,
                       "strided byte layout has zero inner bytes");

  uint64_t segmentCount = 1;
  for (uint32_t iterations : layout.iterations) {
    if (iterations == 0)
      return memoryError(TargetModelMemoryErrorCode::InvalidEffect,
                         "strided byte layout has zero iterations");
    if (iterations > std::numeric_limits<uint64_t>::max() / segmentCount)
      return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                         "strided byte layout segment count overflows");
    segmentCount *= iterations;
  }

  uint64_t payloadBytes = 0;
  if (layout.innerBytes > std::numeric_limits<uint64_t>::max() / segmentCount)
    return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                       "strided byte layout payload size overflows");
  payloadBytes = segmentCount * layout.innerBytes;

  uint64_t boundingSpan = layout.innerBytes;
  for (size_t dimension = 0; dimension < layout.iterations.size();
       ++dimension) {
    const uint64_t iterations = layout.iterations[dimension];
    if (iterations == 1)
      continue;
    const uint64_t coordinate = iterations - 1;
    if (layout.strides[dimension] >
        std::numeric_limits<uint64_t>::max() / coordinate)
      return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                         "strided byte layout address span overflows");
    const uint64_t delta = coordinate * layout.strides[dimension];
    if (!checkedAdd(boundingSpan, delta, boundingSpan))
      return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                         "strided byte layout address span overflows");
  }

  // Descriptor iteration order is independent of physical nesting. Sorting
  // active dimensions by stride recognizes both row-major and transposed
  // layouts: each next stride must begin after the span covered so far.
  std::array<size_t, 3> dimensions{0, 1, 2};
  llvm::sort(dimensions, [&](size_t lhs, size_t rhs) {
    return std::tie(layout.strides[lhs], lhs) <
           std::tie(layout.strides[rhs], rhs);
  });
  uint64_t coveredSpan = layout.innerBytes;
  bool analyticallyNonOverlapping = true;
  for (size_t dimension : dimensions) {
    const uint64_t iterations = layout.iterations[dimension];
    if (iterations == 1)
      continue;
    const uint64_t stride = layout.strides[dimension];
    if (stride < coveredSpan) {
      analyticallyNonOverlapping = false;
      break;
    }
    const uint64_t delta = (iterations - 1) * stride;
    if (!checkedAdd(coveredSpan, delta, coveredSpan))
      return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                         "strided byte layout covered span overflows");
  }

  return StridedLayoutInfo{segmentCount, payloadBytes, boundingSpan,
                           analyticallyNonOverlapping};
}

bool layoutPreservesAlignment(const TargetModelStridedByteLayout &layout,
                              uint64_t requiredAlignment) {
  if (!isPowerOfTwo(requiredAlignment))
    return false;
  for (size_t dimension = 0; dimension < layout.iterations.size(); ++dimension)
    if (layout.iterations[dimension] > 1 &&
        layout.strides[dimension] % requiredAlignment != 0)
      return false;
  return true;
}

template <typename Callback>
llvm::Error forEachStridedSegment(const TargetModelStridedByteLayout &layout,
                                  Callback &&callback) {
  uint64_t payloadOffset = 0;
  for (uint64_t outer = 0; outer < layout.iterations[2]; ++outer)
    for (uint64_t middle = 0; middle < layout.iterations[1]; ++middle)
      for (uint64_t inner = 0; inner < layout.iterations[0]; ++inner) {
        const std::array<uint64_t, 3> coordinates{inner, middle, outer};
        uint64_t addressDelta = 0;
        for (size_t dimension = 0; dimension < coordinates.size();
             ++dimension) {
          if (coordinates[dimension] != 0 &&
              layout.strides[dimension] >
                  std::numeric_limits<uint64_t>::max() / coordinates[dimension])
            return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                               "strided byte layout address overflows");
          const uint64_t delta =
              coordinates[dimension] * layout.strides[dimension];
          if (!checkedAdd(addressDelta, delta, addressDelta))
            return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                               "strided byte layout address overflows");
        }
        if (llvm::Error error = callback(addressDelta, payloadOffset))
          return error;
        if (!checkedAdd(payloadOffset, layout.innerBytes, payloadOffset))
          return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                             "strided byte layout payload offset overflows");
      }
  return llvm::Error::success();
}

struct ResolvedWriteSegment {
  TargetModelResolvedRange range;
  uint64_t payloadOffset = 0;
  std::vector<uint8_t> *backing = nullptr;
};

struct ResolvedWriteInterval {
  int64_t launchSlot = -1;
  TargetModelAddressSpace addressSpace = TargetModelAddressSpace::TileSPM;
  std::optional<TargetModelResourceId> resource;
  uint64_t regionOffset = 0;
  uint64_t byteCount = 0;
};

ResolvedWriteInterval makeInterval(const TargetModelResolvedRange &range,
                                   uint64_t regionOffset, uint64_t byteCount) {
  return {range.launchSlot, range.addressSpace, range.resource, regionOffset,
          byteCount};
}

bool sameResource(const ResolvedWriteInterval &lhs,
                  const ResolvedWriteInterval &rhs) {
  if (lhs.addressSpace != rhs.addressSpace)
    return false;
  if (lhs.addressSpace == TargetModelAddressSpace::TileSPM)
    return lhs.launchSlot == rhs.launchSlot;
  return lhs.resource == rhs.resource;
}

llvm::Error
rejectOverlappingIntervals(std::vector<ResolvedWriteInterval> intervals) {
  for (size_t left = 0; left < intervals.size(); ++left)
    for (size_t right = left + 1; right < intervals.size(); ++right) {
      const ResolvedWriteInterval &lhs = intervals[left];
      const ResolvedWriteInterval &rhs = intervals[right];
      if (!sameResource(lhs, rhs))
        continue;
      uint64_t lhsEnd = 0;
      uint64_t rhsEnd = 0;
      (void)checkedAdd(lhs.regionOffset, lhs.byteCount, lhsEnd);
      (void)checkedAdd(rhs.regionOffset, rhs.byteCount, rhsEnd);
      if (lhs.regionOffset < rhsEnd && rhs.regionOffset < lhsEnd)
        return memoryError(TargetModelMemoryErrorCode::InvalidEffect,
                           "pending writes overlap one private resource");
    }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::vector<uint8_t>>
InvocationMemoryRegistry::readStridedSnapshot(
    int64_t launchSlot, TargetModelAddressSpace addressSpace, uint64_t address,
    const TargetModelStridedByteLayout &layout,
    uint64_t requiredAlignment) const {
  llvm::Expected<StridedLayoutInfo> info = inspectStridedLayout(layout);
  if (!info)
    return info.takeError();
  if (info->payloadBytes > std::numeric_limits<size_t>::max())
    return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                       "strided snapshot exceeds host size_t");
  std::vector<uint8_t> result;
  if (info->payloadBytes > result.max_size())
    return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                       "strided snapshot exceeds host vector capacity");
  uint64_t boundingEnd = 0;
  if (!checkedAdd(address, info->boundingSpan, boundingEnd))
    return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                       "strided snapshot address span overflows");

  auto getBacking = [&](const TargetModelResolvedRange &resolved)
      -> const std::vector<uint8_t> * {
    if (resolved.addressSpace == TargetModelAddressSpace::TileSPM) {
      const TileSPMStorage *storage = findSPM(resolved.launchSlot);
      return storage ? &storage->bytes : nullptr;
    }
    if (!resolved.resource)
      return nullptr;
    const ResourceStorage *storage = findResource(*resolved.resource);
    return storage ? &storage->bytes : nullptr;
  };
  auto validateBacking =
      [&](const TargetModelResolvedRange &resolved,
          const std::vector<uint8_t> *backing) -> llvm::Error {
    if (!backing)
      return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                         "resolved resource has no private backing storage");
    if (resolved.regionOffset > std::numeric_limits<size_t>::max() ||
        resolved.byteCount > std::numeric_limits<size_t>::max())
      return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                         "resolved strided snapshot exceeds host size_t");
    const size_t offset = static_cast<size_t>(resolved.regionOffset);
    const size_t byteCount = static_cast<size_t>(resolved.byteCount);
    if (offset > backing->size() || byteCount > backing->size() - offset)
      return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                         "resolved strided snapshot exceeds private storage");
    return llvm::Error::success();
  };

  std::optional<TargetModelResolvedRange> fastRange;
  const std::vector<uint8_t> *fastBacking = nullptr;
  if (layoutPreservesAlignment(layout, requiredAlignment)) {
    llvm::Expected<TargetModelResolvedRange> bounding =
        plan.resolve(launchSlot, addressSpace, TargetModelAccess::Read, address,
                     info->boundingSpan, requiredAlignment);
    if (bounding) {
      fastBacking = getBacking(*bounding);
      if (llvm::Error error = validateBacking(*bounding, fastBacking))
        return std::move(error);
      fastRange = *bounding;
    } else {
      llvm::consumeError(bounding.takeError());
    }
  }

  result.resize(static_cast<size_t>(info->payloadBytes));
  auto copyFromBacking = [&](const std::vector<uint8_t> &backing,
                             uint64_t regionOffset,
                             uint64_t payloadOffset) -> llvm::Error {
    if (regionOffset > std::numeric_limits<size_t>::max() ||
        payloadOffset > std::numeric_limits<size_t>::max())
      return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                         "strided snapshot offset exceeds host size_t");
    const size_t source = static_cast<size_t>(regionOffset);
    const size_t destination = static_cast<size_t>(payloadOffset);
    const size_t byteCount = layout.innerBytes;
    if (source > backing.size() || byteCount > backing.size() - source ||
        destination > result.size() || byteCount > result.size() - destination)
      return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                         "resolved strided snapshot exceeds private storage");
    std::copy(backing.begin() + source, backing.begin() + source + byteCount,
              result.begin() + destination);
    return llvm::Error::success();
  };
  if (fastRange) {
    if (llvm::Error error = forEachStridedSegment(
            layout, [&](uint64_t addressDelta, uint64_t payloadOffset) {
              uint64_t regionOffset = 0;
              if (!checkedAdd(fastRange->regionOffset, addressDelta,
                              regionOffset))
                return memoryError(
                    TargetModelMemoryErrorCode::AddressOverflow,
                    "resolved strided snapshot offset overflows");
              return copyFromBacking(*fastBacking, regionOffset, payloadOffset);
            }))
      return std::move(error);
    return result;
  }

  std::optional<TargetModelResolvedRange> cachedResource;
  const std::vector<uint8_t> *cachedBacking = nullptr;
  if (llvm::Error error = forEachStridedSegment(
          layout, [&](uint64_t addressDelta, uint64_t payloadOffset) {
            uint64_t segmentAddress = 0;
            if (!checkedAdd(address, addressDelta, segmentAddress))
              return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                                 "strided snapshot address overflows");
            llvm::Expected<TargetModelResolvedRange> resolved = plan.resolve(
                launchSlot, addressSpace, TargetModelAccess::Read,
                segmentAddress, layout.innerBytes, requiredAlignment);
            if (!resolved)
              return resolved.takeError();
            if (!cachedResource ||
                cachedResource->launchSlot != resolved->launchSlot ||
                cachedResource->addressSpace != resolved->addressSpace ||
                cachedResource->resource != resolved->resource) {
              cachedBacking = getBacking(*resolved);
              if (llvm::Error validation =
                      validateBacking(*resolved, cachedBacking))
                return validation;
              cachedResource = *resolved;
            }
            return copyFromBacking(*cachedBacking, resolved->regionOffset,
                                   payloadOffset);
          }))
    return std::move(error);
  return result;
}

llvm::Error InvocationMemoryRegistry::applyAtomically(
    llvm::ArrayRef<TargetModelByteWrite> pendingWrites) {
  struct ValidatedWrite {
    const TargetModelByteWrite *write = nullptr;
    std::optional<TargetModelResolvedRange> contiguousOrBounding;
    std::vector<ResolvedWriteSegment> fallbackSegments;
    std::vector<uint8_t> *backing = nullptr;
  };

  std::vector<ValidatedWrite> validatedWrites;
  validatedWrites.reserve(pendingWrites.size());
  for (const TargetModelByteWrite &write : pendingWrites) {
    if (write.bytes.empty())
      return memoryError(TargetModelMemoryErrorCode::InvalidEffect,
                         "pending write has no bytes");
    ValidatedWrite validated;
    validated.write = &write;
    if (!write.stridedLayout) {
      llvm::Expected<TargetModelResolvedRange> resolved = plan.resolve(
          write.launchSlot, write.addressSpace, TargetModelAccess::Write,
          write.address, write.bytes.size(), write.requiredAlignment);
      if (!resolved)
        return resolved.takeError();
      validated.contiguousOrBounding = *resolved;
      validatedWrites.push_back(std::move(validated));
      continue;
    }

    llvm::Expected<StridedLayoutInfo> info =
        inspectStridedLayout(*write.stridedLayout);
    if (!info)
      return info.takeError();
    if (info->payloadBytes != write.bytes.size())
      return memoryError(
          TargetModelMemoryErrorCode::InvalidEffect,
          "strided pending write payload differs from descriptor byte count");
    uint64_t boundingEnd = 0;
    if (!checkedAdd(write.address, info->boundingSpan, boundingEnd))
      return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                         "strided pending write address span overflows");
    if (info->analyticallyNonOverlapping &&
        layoutPreservesAlignment(*write.stridedLayout,
                                 write.requiredAlignment)) {
      llvm::Expected<TargetModelResolvedRange> bounding = plan.resolve(
          write.launchSlot, write.addressSpace, TargetModelAccess::Write,
          write.address, info->boundingSpan, write.requiredAlignment);
      if (bounding)
        validated.contiguousOrBounding = *bounding;
      else
        llvm::consumeError(bounding.takeError());
    }

    if (!validated.contiguousOrBounding) {
      if (info->segmentCount > std::numeric_limits<size_t>::max() ||
          info->segmentCount > validated.fallbackSegments.max_size())
        return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                           "strided pending write has too many segments");
      validated.fallbackSegments.reserve(
          static_cast<size_t>(info->segmentCount));
      if (llvm::Error error = forEachStridedSegment(
              *write.stridedLayout,
              [&](uint64_t addressDelta,
                  uint64_t payloadOffset) -> llvm::Error {
                uint64_t segmentAddress = 0;
                if (!checkedAdd(write.address, addressDelta, segmentAddress))
                  return memoryError(
                      TargetModelMemoryErrorCode::AddressOverflow,
                      "strided pending write address overflows");
                llvm::Expected<TargetModelResolvedRange> resolved =
                    plan.resolve(write.launchSlot, write.addressSpace,
                                 TargetModelAccess::Write, segmentAddress,
                                 write.stridedLayout->innerBytes,
                                 write.requiredAlignment);
                if (!resolved)
                  return resolved.takeError();
                validated.fallbackSegments.push_back(
                    {*resolved, payloadOffset, nullptr});
                return llvm::Error::success();
              }))
        return error;
    }
    validatedWrites.push_back(std::move(validated));
  }

  std::vector<ResolvedWriteInterval> intervals;
  auto appendIntervals = [&](const ValidatedWrite &validated) -> llvm::Error {
    const TargetModelByteWrite &write = *validated.write;
    if (!write.stridedLayout) {
      const TargetModelResolvedRange &resolved =
          *validated.contiguousOrBounding;
      intervals.push_back(
          makeInterval(resolved, resolved.regionOffset, resolved.byteCount));
      return llvm::Error::success();
    }
    if (!validated.fallbackSegments.empty()) {
      for (const ResolvedWriteSegment &segment : validated.fallbackSegments)
        intervals.push_back(makeInterval(segment.range,
                                         segment.range.regionOffset,
                                         segment.range.byteCount));
      return llvm::Error::success();
    }
    const TargetModelResolvedRange &bounding = *validated.contiguousOrBounding;
    return forEachStridedSegment(
        *write.stridedLayout,
        [&](uint64_t addressDelta, uint64_t) -> llvm::Error {
          uint64_t regionOffset = 0;
          if (!checkedAdd(bounding.regionOffset, addressDelta, regionOffset))
            return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                               "resolved pending write offset overflows");
          intervals.push_back(makeInterval(bounding, regionOffset,
                                           write.stridedLayout->innerBytes));
          return llvm::Error::success();
        });
  };

  if (validatedWrites.size() > 1) {
    for (const ValidatedWrite &validated : validatedWrites)
      if (llvm::Error error = appendIntervals(validated))
        return error;
  } else if (!validatedWrites.empty() &&
             !validatedWrites.front().fallbackSegments.empty()) {
    if (llvm::Error error = appendIntervals(validatedWrites.front()))
      return error;
  }
  if (llvm::Error error = rejectOverlappingIntervals(std::move(intervals)))
    return error;

  auto getBacking =
      [&](const TargetModelResolvedRange &resolved) -> std::vector<uint8_t> * {
    if (resolved.addressSpace == TargetModelAddressSpace::TileSPM) {
      TileSPMStorage *storage = findSPM(resolved.launchSlot);
      return storage ? &storage->bytes : nullptr;
    }
    if (!resolved.resource)
      return nullptr;
    ResourceStorage *storage = findResource(*resolved.resource);
    return storage ? &storage->bytes : nullptr;
  };
  auto validateBacking = [&](const TargetModelResolvedRange &resolved,
                             std::vector<uint8_t> *backing) -> llvm::Error {
    if (!backing)
      return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                         "resolved resource has no private backing storage");
    if (resolved.regionOffset > std::numeric_limits<size_t>::max() ||
        resolved.byteCount > std::numeric_limits<size_t>::max())
      return memoryError(TargetModelMemoryErrorCode::AddressOverflow,
                         "resolved pending write exceeds host size_t");
    const size_t offset = static_cast<size_t>(resolved.regionOffset);
    const size_t byteCount = static_cast<size_t>(resolved.byteCount);
    if (offset > backing->size() || byteCount > backing->size() - offset)
      return memoryError(TargetModelMemoryErrorCode::InvalidInvocation,
                         "resolved pending write exceeds private storage");
    return llvm::Error::success();
  };
  for (ValidatedWrite &validated : validatedWrites) {
    if (validated.contiguousOrBounding) {
      validated.backing = getBacking(*validated.contiguousOrBounding);
      if (llvm::Error error = validateBacking(*validated.contiguousOrBounding,
                                              validated.backing))
        return error;
      continue;
    }
    for (ResolvedWriteSegment &segment : validated.fallbackSegments) {
      segment.backing = getBacking(segment.range);
      if (llvm::Error error = validateBacking(segment.range, segment.backing))
        return error;
    }
  }

  auto applySegment = [&](const TargetModelByteWrite &write,
                          std::vector<uint8_t> &backing, uint64_t regionOffset,
                          uint64_t payloadOffset, uint64_t byteCount) {
    const size_t destination = static_cast<size_t>(regionOffset);
    const size_t source = static_cast<size_t>(payloadOffset);
    const size_t size = static_cast<size_t>(byteCount);
    std::copy(write.bytes.begin() + source, write.bytes.begin() + source + size,
              backing.begin() + destination);
  };

  for (const ValidatedWrite &validated : validatedWrites) {
    const TargetModelByteWrite &write = *validated.write;
    if (!write.stridedLayout) {
      const TargetModelResolvedRange &resolved =
          *validated.contiguousOrBounding;
      applySegment(write, *validated.backing, resolved.regionOffset, 0,
                   resolved.byteCount);
      continue;
    }
    if (!validated.fallbackSegments.empty()) {
      for (const ResolvedWriteSegment &segment : validated.fallbackSegments)
        applySegment(write, *segment.backing, segment.range.regionOffset,
                     segment.payloadOffset, segment.range.byteCount);
      continue;
    }
    const TargetModelResolvedRange &bounding = *validated.contiguousOrBounding;
    if (llvm::Error error = forEachStridedSegment(
            *write.stridedLayout,
            [&](uint64_t addressDelta, uint64_t payloadOffset) {
              uint64_t regionOffset = 0;
              if (!checkedAdd(bounding.regionOffset, addressDelta,
                              regionOffset))
                llvm_unreachable("validated strided offset must not overflow");
              applySegment(write, *validated.backing, regionOffset,
                           payloadOffset, write.stridedLayout->innerBytes);
              return llvm::Error::success();
            }))
      llvm_unreachable("validated strided layout must remain iterable");
  }
  return llvm::Error::success();
}

} // namespace wafer::model
