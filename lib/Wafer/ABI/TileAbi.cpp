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

static bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

static bool checkedRangeEndInclusive(uint64_t base, uint64_t span,
                                     uint64_t &end) {
  if (span == 0)
    return false;
  uint64_t exclusiveEnd = 0;
  if (!checkedAdd(base, span, exclusiveEnd))
    return false;
  end = exclusiveEnd - 1;
  return true;
}

static bool checkedU32(uint64_t value, const char *message, uint32_t &result,
                       std::string *error) {
  if (value > std::numeric_limits<uint32_t>::max())
    return fail(error, message);
  result = static_cast<uint32_t>(value);
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

static bool computeIterationProduct(std::array<int64_t, 3> iterations,
                                    uint64_t &product, std::string *error) {
  product = 1;
  for (int64_t iteration : iterations) {
    if (iteration <= 0)
      return fail(error, "DMA iterations must be positive");
    uint64_t next = 0;
    if (!checkedMul(product, static_cast<uint64_t>(iteration), next))
      return fail(error, "DMA iteration product overflows");
    product = next;
  }
  return true;
}

static bool encodeStrideIteration(std::array<int64_t, 3> strides,
                                  std::array<int64_t, 3> iterations,
                                  std::array<uint32_t, 3> &encodedStrides,
                                  std::array<uint32_t, 3> &encodedIterations,
                                  std::string *error) {
  for (int index = 0; index < 3; ++index) {
    if (strides[index] < 0)
      return fail(error, "DMA strides must be non-negative");
    if (!checkedU32(static_cast<uint64_t>(strides[index]),
                    "DMA stride is not representable", encodedStrides[index],
                    error))
      return false;
    if (iterations[index] <= 0)
      return fail(error, "DMA iterations must be positive");
    uint64_t registerIteration = static_cast<uint64_t>(iterations[index] - 1);
    if (!checkedU32(registerIteration, "DMA iteration is not representable",
                    encodedIterations[index], error))
      return false;
  }
  return true;
}

static bool getFormatElementBytes(DataFormat format, uint64_t &elementBytes,
                                  bool &bitPacked) {
  bitPacked = false;
  switch (format) {
  case DataFormat::INT8:
  case DataFormat::UINT8:
    elementBytes = 1;
    return true;
  case DataFormat::INT16:
  case DataFormat::FP16:
  case DataFormat::BF16:
  case DataFormat::UINT16:
    elementBytes = 2;
    return true;
  case DataFormat::INT32:
  case DataFormat::FP32:
  case DataFormat::TF32:
  case DataFormat::UINT32:
    elementBytes = 4;
    return true;
  case DataFormat::INT64:
  case DataFormat::UINT64:
    elementBytes = 8;
    return true;
  case DataFormat::BOOL:
    elementBytes = 1;
    bitPacked = true;
    return true;
  }
  return false;
}

static bool computeInnerElemCount(uint64_t innerBytes, DataFormat format,
                                  uint32_t &elemCount, std::string *error) {
  uint64_t elementBytes = 0;
  bool bitPacked = false;
  if (!getFormatElementBytes(format, elementBytes, bitPacked))
    return fail(error, "unsupported Data_Format");

  uint64_t count = 0;
  if (bitPacked) {
    if (!checkedMul(innerBytes, uint64_t{8}, count))
      return fail(error, "DMA element count overflows");
  } else {
    if (innerBytes % elementBytes != 0)
      return fail(error,
                  "DMA inner byte count must be divisible by format element "
                  "bytes");
    count = innerBytes / elementBytes;
  }
  return checkedU32(count, "DMA element count is not representable", elemCount,
                    error);
}

static bool validateSpmOffset(uint64_t offset, const char *message,
                              std::string *error) {
  if (offset > kUsableSpmBytes)
    return fail(error, message);
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
                /*strides=*/{0, 0, 0}, /*iterations=*/{1, 1, 1}, fullDescriptor,
                error))
    return false;

  descriptor = Dma1DDescriptor{
      fullDescriptor.ddrAddress,      fullDescriptor.spmOffset,
      fullDescriptor.byteCount,       fullDescriptor.ddrEndExclusive,
      fullDescriptor.spmEndExclusive, fullDescriptor.waitPolicy};
  return true;
}

static bool buildDmaRegisterPacket(const DmaDescriptor &descriptor,
                                   DataFormat format, bool isRdma,
                                   DmaRegisterPacket &packet,
                                   std::string *error) {
  std::array<uint32_t, 3> encodedStrides;
  std::array<uint32_t, 3> encodedIterations;
  if (!encodeStrideIteration(descriptor.strides, descriptor.iterations,
                             encodedStrides, encodedIterations, error))
    return false;

  uint32_t elemCount = 0;
  if (!computeInnerElemCount(descriptor.innerBytes, format, elemCount, error))
    return false;

  uint64_t stridedSpan = 0;
  if (!computeStridedSpan(descriptor.strides, descriptor.iterations,
                          descriptor.innerBytes, stridedSpan, error))
    return false;

  uint64_t src = isRdma ? descriptor.ddrAddress : descriptor.spmOffset;
  uint64_t dst = isRdma ? descriptor.spmOffset : descriptor.ddrAddress;
  uint64_t srcSpan = isRdma ? stridedSpan : descriptor.byteCount;
  uint64_t dstSpan = isRdma ? descriptor.byteCount : stridedSpan;

  uint64_t srcEnd = 0;
  uint64_t dstEnd = 0;
  if (!checkedRangeEndInclusive(src, srcSpan, srcEnd) ||
      !checkedRangeEndInclusive(dst, dstSpan, dstEnd))
    return fail(error, "DMA range end overflows");

  packet = DmaRegisterPacket{isRdma ? kInterTypeRDMA : kInterTypeWDMA,
                             src,
                             dst,
                             encodedStrides[0],
                             encodedIterations[0],
                             encodedStrides[1],
                             encodedIterations[1],
                             encodedStrides[2],
                             encodedIterations[2],
                             elemCount,
                             format,
                             srcEnd,
                             dstEnd,
                             descriptor.waitPolicy};
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

bool buildGatherScatter(uint32_t spmSrcOffset, uint32_t spmDstOffset,
                        uint64_t byteCount, uint64_t innerBytes,
                        std::array<int64_t, 3> srcStrides,
                        std::array<int64_t, 3> srcIterations,
                        std::array<int64_t, 3> dstStrides,
                        std::array<int64_t, 3> dstIterations,
                        GatherScatterDescriptor &descriptor,
                        std::string *error) {
  if (byteCount == 0)
    return fail(error, "gather/scatter byte count must be positive");
  if (innerBytes == 0)
    return fail(error, "gather/scatter inner byte count must be positive");
  if (innerBytes > byteCount)
    return fail(error,
                "gather/scatter inner byte count must not exceed byte count");
  if (!validateSpmOffset(spmSrcOffset,
                         "gather/scatter source SPM offset out of range",
                         error) ||
      !validateSpmOffset(spmDstOffset,
                         "gather/scatter dest SPM offset out of range", error))
    return false;

  uint64_t srcSpan = 0;
  if (!computeStridedSpan(srcStrides, srcIterations, innerBytes, srcSpan,
                          error))
    return false;
  uint64_t dstSpan = 0;
  if (!computeStridedSpan(dstStrides, dstIterations, innerBytes, dstSpan,
                          error))
    return false;

  uint64_t srcProduct = 0;
  if (!computeIterationProduct(srcIterations, srcProduct, error))
    return false;
  uint64_t dstProduct = 0;
  if (!computeIterationProduct(dstIterations, dstProduct, error))
    return false;
  if (srcProduct != dstProduct)
    return fail(error,
                "gather/scatter source and dest iteration products must match");
  uint64_t movedBytes = 0;
  if (!checkedMul(innerBytes, srcProduct, movedBytes))
    return fail(error, "gather/scatter byte count overflows");
  if (movedBytes != byteCount)
    return fail(error, "gather/scatter byte count must equal inner bytes times "
                       "iteration product");

  uint64_t srcEnd = 0;
  if (!checkedAdd(spmSrcOffset, srcSpan, srcEnd))
    return fail(error, "gather/scatter source range end overflows");
  uint64_t dstEnd = 0;
  if (!checkedAdd(spmDstOffset, dstSpan, dstEnd))
    return fail(error, "gather/scatter dest range end overflows");
  if (srcEnd > kUsableSpmBytes)
    return fail(error, "gather/scatter source range exceeds usable capacity");
  if (dstEnd > kUsableSpmBytes)
    return fail(error, "gather/scatter dest range exceeds usable capacity");

  uint32_t srcEndU32 = 0;
  uint32_t dstEndU32 = 0;
  if (!checkedU32(srcEnd,
                  "gather/scatter source range end is not representable",
                  srcEndU32, error) ||
      !checkedU32(dstEnd, "gather/scatter dest range end is not representable",
                  dstEndU32, error))
    return false;

  descriptor = GatherScatterDescriptor{
      spmSrcOffset,         spmDstOffset, byteCount,     innerBytes, srcStrides,
      srcIterations,        dstStrides,   dstIterations, srcEndU32,  dstEndU32,
      WaitPolicy::IssueOnly};
  if (error)
    error->clear();
  return true;
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

bool buildLocalFence(LocalFenceDescriptor &descriptor, std::string *error) {
  descriptor = LocalFenceDescriptor{WaitPolicy::LocalWait};
  if (error)
    error->clear();
  return true;
}

bool buildRdmaRegisterPacket(const DmaDescriptor &descriptor, DataFormat format,
                             DmaRegisterPacket &packet, std::string *error) {
  return buildDmaRegisterPacket(descriptor, format, /*isRdma=*/true, packet,
                                error);
}

bool buildWdmaRegisterPacket(const DmaDescriptor &descriptor, DataFormat format,
                             DmaRegisterPacket &packet, std::string *error) {
  return buildDmaRegisterPacket(descriptor, format, /*isRdma=*/false, packet,
                                error);
}

bool buildGatherScatterRegisterPacket(const GatherScatterDescriptor &descriptor,
                                      GatherScatterRegisterPacket &packet,
                                      std::string *error) {
  std::array<uint32_t, 3> encodedSrcStrides;
  std::array<uint32_t, 3> encodedSrcIterations;
  if (!encodeStrideIteration(descriptor.srcStrides, descriptor.srcIterations,
                             encodedSrcStrides, encodedSrcIterations, error))
    return false;

  std::array<uint32_t, 3> encodedDstStrides;
  std::array<uint32_t, 3> encodedDstIterations;
  if (!encodeStrideIteration(descriptor.dstStrides, descriptor.dstIterations,
                             encodedDstStrides, encodedDstIterations, error))
    return false;

  uint32_t innerBytes = 0;
  if (!checkedU32(descriptor.innerBytes,
                  "gather/scatter inner byte count is not representable",
                  innerBytes, error))
    return false;

  uint64_t srcSpan = 0;
  if (!computeStridedSpan(descriptor.srcStrides, descriptor.srcIterations,
                          descriptor.innerBytes, srcSpan, error))
    return false;
  uint64_t dstSpan = 0;
  if (!computeStridedSpan(descriptor.dstStrides, descriptor.dstIterations,
                          descriptor.innerBytes, dstSpan, error))
    return false;

  uint64_t srcEnd = 0;
  uint64_t dstEnd = 0;
  if (!checkedRangeEndInclusive(descriptor.spmSrcOffset, srcSpan, srcEnd) ||
      !checkedRangeEndInclusive(descriptor.spmDstOffset, dstSpan, dstEnd))
    return fail(error, "gather/scatter range end overflows");

  uint32_t srcEndU32 = 0;
  uint32_t dstEndU32 = 0;
  if (!checkedU32(srcEnd, "gather/scatter source end is not representable",
                  srcEndU32, error) ||
      !checkedU32(dstEnd, "gather/scatter dest end is not representable",
                  dstEndU32, error))
    return false;

  packet = GatherScatterRegisterPacket{kInterTypeTDMA,
                                       kTdmaGatherScatterOpcode,
                                       descriptor.spmSrcOffset,
                                       descriptor.spmDstOffset,
                                       innerBytes,
                                       encodedSrcStrides[0],
                                       encodedSrcIterations[0],
                                       encodedSrcStrides[1],
                                       encodedSrcIterations[1],
                                       encodedSrcStrides[2],
                                       encodedSrcIterations[2],
                                       encodedDstStrides[0],
                                       encodedDstIterations[0],
                                       encodedDstStrides[1],
                                       encodedDstIterations[1],
                                       encodedDstStrides[2],
                                       encodedDstIterations[2],
                                       srcEndU32,
                                       dstEndU32,
                                       descriptor.waitPolicy};
  if (error)
    error->clear();
  return true;
}

bool buildGemmRegisterPacket(const GemmDescriptor &descriptor,
                             uint32_t lhsSpmOffset, uint32_t rhsSpmOffset,
                             uint32_t destSpmOffset, DataFormat inputFormat,
                             DataFormat outputFormat,
                             GemmRegisterPacket &packet, std::string *error) {
  if (!validateSpmOffset(lhsSpmOffset, "GEMM lhs SPM offset out of range",
                         error) ||
      !validateSpmOffset(rhsSpmOffset, "GEMM rhs SPM offset out of range",
                         error) ||
      !validateSpmOffset(destSpmOffset, "GEMM dest SPM offset out of range",
                         error))
    return false;
  if (descriptor.m > std::numeric_limits<uint16_t>::max() ||
      descriptor.k > std::numeric_limits<uint16_t>::max() ||
      descriptor.n > std::numeric_limits<uint16_t>::max())
    return fail(error, "GEMM dimensions must fit hardware 16-bit fields");

  packet = GemmRegisterPacket{kInterTypeNEUR,
                              kNeTypeGemm,
                              lhsSpmOffset,
                              rhsSpmOffset,
                              destSpmOffset,
                              static_cast<uint32_t>(descriptor.m),
                              static_cast<uint32_t>(descriptor.k),
                              static_cast<uint32_t>(descriptor.n),
                              1,
                              1,
                              0,
                              0,
                              inputFormat,
                              outputFormat,
                              false,
                              descriptor.waitPolicy};
  if (error)
    error->clear();
  return true;
}

bool buildLocalFenceRegisterCall(const LocalFenceDescriptor &descriptor,
                                 LocalFenceRegisterCall &call,
                                 std::string *error) {
  if (descriptor.waitPolicy != WaitPolicy::LocalWait)
    return fail(error, "local fence must use local_wait policy");

  call = LocalFenceRegisterCall{/*callsLocalWait=*/true,
                                /*waitPolicy=*/WaitPolicy::LocalWait};
  if (error)
    error->clear();
  return true;
}

} // namespace abi
} // namespace wafer
