//===- TileAbi.cpp - Tile C ABI descriptor builders --------------------------===//

#include "Wafer/ABI/TileAbi.h"

#include <algorithm>
#include <limits>

namespace wafer {
namespace abi {
namespace {

static bool fail(std::string *error, const char *message) {
  if (error)
    *error = message;
  return false;
}

static bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedMulAdd(uint64_t lhs, uint64_t rhs, uint64_t addend,
                          uint64_t &result) {
  if (lhs != 0 && rhs > (std::numeric_limits<uint64_t>::max() - addend) / lhs)
    return false;
  result = lhs * rhs + addend;
  return true;
}

static bool computeStridedSpan(std::array<int64_t, 3> strides,
                               std::array<int64_t, 3> iterations,
                               uint64_t innerBytes, uint64_t &span,
                               std::string *error) {
  uint64_t maxOffset = 0;
  for (int index = 0; index < 3; ++index) {
    if (iterations[index] <= 0)
      return fail(error, "DMA iterations must be positive");
    if (strides[index] < 0)
      return fail(error, "DMA strides must be non-negative");
    uint64_t contribution = 0;
    if (!checkedMulAdd(static_cast<uint64_t>(iterations[index] - 1),
                       static_cast<uint64_t>(strides[index]), maxOffset,
                       contribution))
      return fail(error, "DMA strided range end overflows");
    maxOffset = contribution;
  }
  if (!checkedAdd(maxOffset, innerBytes, span))
    return fail(error, "DMA strided range end overflows");
  return true;
}

static bool buildDma(uint64_t ddrAddress, uint32_t spmOffset,
                     uint64_t byteCount, uint64_t innerBytes,
                     std::array<int64_t, 3> strides,
                     std::array<int64_t, 3> iterations,
                     DmaDescriptor &descriptor, std::string *error) {
  if (byteCount == 0)
    return fail(error, "DMA byte count must be positive");
  if (innerBytes == 0)
    return fail(error, "DMA inner byte count must be positive");
  if (innerBytes > byteCount)
    return fail(error, "DMA inner byte count must not exceed byte count");
  if (ddrAddress < kDdrLowerBound)
    return fail(error, "DDR address below lower bound");

  uint64_t stridedSpan = 0;
  if (!computeStridedSpan(strides, iterations, innerBytes, stridedSpan, error))
    return false;

  uint64_t ddrEnd = 0;
  if (!checkedAdd(ddrAddress, std::max(byteCount, stridedSpan), ddrEnd))
    return fail(error, "DDR range end overflows");

  uint64_t spmEnd = 0;
  if (!checkedAdd(spmOffset, byteCount, spmEnd))
    return fail(error, "SPM range end overflows");
  if (spmEnd > kUsableSpmBytes)
    return fail(error, "SPM range exceeds usable capacity");
  if (spmEnd > std::numeric_limits<uint32_t>::max())
    return fail(error, "SPM range end is not representable");

  descriptor = DmaDescriptor{ddrAddress,
                             spmOffset,
                             byteCount,
                             innerBytes,
                             strides,
                             iterations,
                             ddrEnd,
                             static_cast<uint32_t>(spmEnd),
                             WaitPolicy::IssueOnly};
  if (error)
    error->clear();
  return true;
}

static bool buildDma1D(uint64_t ddrAddress, uint32_t spmOffset, uint64_t bytes,
                       Dma1DDescriptor &descriptor, std::string *error) {
  DmaDescriptor fullDescriptor;
  if (!buildDma(ddrAddress, spmOffset, bytes, bytes,
                /*strides=*/{0, 0, 0}, /*iterations=*/{1, 1, 1},
                fullDescriptor, error))
    return false;

  descriptor = Dma1DDescriptor{fullDescriptor.ddrAddress,
                               fullDescriptor.spmOffset,
                               fullDescriptor.byteCount,
                               fullDescriptor.ddrEndExclusive,
                               fullDescriptor.spmEndExclusive,
                               fullDescriptor.waitPolicy};
  return true;
}

} // namespace

bool buildRdma1D(uint64_t ddrSrcAddr, uint32_t spmDstOffset, uint64_t bytes,
                 Dma1DDescriptor &descriptor, std::string *error) {
  return buildDma1D(ddrSrcAddr, spmDstOffset, bytes, descriptor, error);
}

bool buildWdma1D(uint32_t spmSrcOffset, uint64_t ddrDstAddr, uint64_t bytes,
                 Dma1DDescriptor &descriptor, std::string *error) {
  return buildDma1D(ddrDstAddr, spmSrcOffset, bytes, descriptor, error);
}

bool buildRdma(uint64_t ddrSrcAddr, uint32_t spmDstOffset, uint64_t byteCount,
               uint64_t innerBytes, std::array<int64_t, 3> strides,
               std::array<int64_t, 3> iterations, DmaDescriptor &descriptor,
               std::string *error) {
  return buildDma(ddrSrcAddr, spmDstOffset, byteCount, innerBytes, strides,
                  iterations, descriptor, error);
}

bool buildWdma(uint32_t spmSrcOffset, uint64_t ddrDstAddr, uint64_t byteCount,
               uint64_t innerBytes, std::array<int64_t, 3> strides,
               std::array<int64_t, 3> iterations, DmaDescriptor &descriptor,
               std::string *error) {
  return buildDma(ddrDstAddr, spmSrcOffset, byteCount, innerBytes, strides,
                  iterations, descriptor, error);
}

bool buildGemm(int64_t m, int64_t k, int64_t n, GemmDescriptor &descriptor,
               std::string *error) {
  if (m <= 0 || k <= 0 || n <= 0)
    return fail(error, "GEMM dimensions must be positive");

  descriptor = GemmDescriptor{m, k, n, WaitPolicy::IssueOnly};
  if (error)
    error->clear();
  return true;
}

} // namespace abi
} // namespace wafer
