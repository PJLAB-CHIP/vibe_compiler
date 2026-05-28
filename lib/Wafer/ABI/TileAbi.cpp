//===- TileAbi.cpp - Tile C ABI descriptor builders --------------------------===//

#include "Wafer/ABI/TileAbi.h"

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

static bool buildDma1D(uint64_t ddrAddress, uint32_t spmOffset, uint64_t bytes,
                       Dma1DDescriptor &descriptor, std::string *error) {
  if (bytes == 0)
    return fail(error, "DMA byte count must be positive");
  if (ddrAddress < kDdrLowerBound)
    return fail(error, "DDR address below lower bound");

  uint64_t ddrEnd = 0;
  if (!checkedAdd(ddrAddress, bytes, ddrEnd))
    return fail(error, "DDR range end overflows");

  uint64_t spmEnd = 0;
  if (!checkedAdd(spmOffset, bytes, spmEnd))
    return fail(error, "SPM range end overflows");
  if (spmEnd > kUsableSpmBytes)
    return fail(error, "SPM range exceeds usable capacity");
  if (spmEnd > std::numeric_limits<uint32_t>::max())
    return fail(error, "SPM range end is not representable");

  descriptor = Dma1DDescriptor{ddrAddress, spmOffset, bytes, ddrEnd,
                               static_cast<uint32_t>(spmEnd),
                               WaitPolicy::IssueOnly};
  if (error)
    error->clear();
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
