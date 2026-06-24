//===- TileAbi.h - Tile C ABI descriptor builders -----------------*- C++ -*-===//

#ifndef WAFER_ABI_TILEABI_H
#define WAFER_ABI_TILEABI_H

#include <array>
#include <cstdint>
#include <string>

namespace wafer {
namespace abi {

constexpr uint64_t kDdrLowerBound = 0x280000000ULL;
constexpr uint64_t kUsableSpmBytes = 0x2f0000ULL;

constexpr uint32_t kInterTypeCGRA = 0;
constexpr uint32_t kInterTypeNEUR = 1;
constexpr uint32_t kInterTypeRDMA = 2;
constexpr uint32_t kInterTypeWDMA = 3;
constexpr uint32_t kInterTypeTDMA = 4;
constexpr uint32_t kNeTypeGemm = 3;
constexpr uint32_t kTdmaGatherScatterOpcode = 135;

enum class WaitPolicy {
  IssueOnly,
};

enum class DataFormat : uint32_t {
  INT8 = 0,
  INT16 = 1,
  FP16 = 2,
  BF16 = 3,
  INT32 = 4,
  FP32 = 5,
  TF32 = 6,
  BOOL = 7,
  UINT8 = 8,
  UINT16 = 9,
  UINT32 = 10,
  INT64 = 11,
  UINT64 = 12,
};

struct Dma1DDescriptor {
  uint64_t ddrAddress = 0;
  uint32_t spmOffset = 0;
  uint64_t bytes = 0;
  uint64_t ddrEndExclusive = 0;
  uint32_t spmEndExclusive = 0;
  WaitPolicy waitPolicy = WaitPolicy::IssueOnly;
};

struct GemmDescriptor {
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
  WaitPolicy waitPolicy = WaitPolicy::IssueOnly;
};

struct DmaDescriptor {
  uint64_t ddrAddress = 0;
  uint32_t spmOffset = 0;
  uint64_t byteCount = 0;
  uint64_t innerBytes = 0;
  std::array<int64_t, 3> strides = {0, 0, 0};
  std::array<int64_t, 3> iterations = {1, 1, 1};
  uint64_t ddrEndExclusive = 0;
  uint32_t spmEndExclusive = 0;
  WaitPolicy waitPolicy = WaitPolicy::IssueOnly;
};

struct GatherScatterDescriptor {
  uint32_t spmSrcOffset = 0;
  uint32_t spmDstOffset = 0;
  uint64_t byteCount = 0;
  uint64_t innerBytes = 0;
  std::array<int64_t, 3> srcStrides = {0, 0, 0};
  std::array<int64_t, 3> srcIterations = {1, 1, 1};
  std::array<int64_t, 3> dstStrides = {0, 0, 0};
  std::array<int64_t, 3> dstIterations = {1, 1, 1};
  uint32_t srcEndExclusive = 0;
  uint32_t dstEndExclusive = 0;
  WaitPolicy waitPolicy = WaitPolicy::IssueOnly;
};

struct DmaRegisterPacket {
  uint32_t interType = 0;
  uint64_t src = 0;
  uint64_t dst = 0;
  uint32_t stride0 = 0;
  uint32_t iteration0 = 0;
  uint32_t stride1 = 0;
  uint32_t iteration1 = 0;
  uint32_t stride2 = 0;
  uint32_t iteration2 = 0;
  uint32_t elemCount = 0;
  DataFormat format = DataFormat::INT8;
  uint64_t srcEnd = 0;
  uint64_t dstEnd = 0;
  WaitPolicy waitPolicy = WaitPolicy::IssueOnly;
};

struct GatherScatterRegisterPacket {
  uint32_t interType = 0;
  uint32_t opcode = 0;
  uint32_t src0 = 0;
  uint32_t dst = 0;
  uint32_t elemCount = 0;
  uint32_t srcStride0 = 0;
  uint32_t srcIteration0 = 0;
  uint32_t srcStride1 = 0;
  uint32_t srcIteration1 = 0;
  uint32_t srcStride2 = 0;
  uint32_t srcIteration2 = 0;
  uint32_t dstStride0 = 0;
  uint32_t dstIteration0 = 0;
  uint32_t dstStride1 = 0;
  uint32_t dstIteration1 = 0;
  uint32_t dstStride2 = 0;
  uint32_t dstIteration2 = 0;
  uint32_t src0End = 0;
  uint32_t dstEnd = 0;
  WaitPolicy waitPolicy = WaitPolicy::IssueOnly;
};

struct GemmRegisterPacket {
  uint32_t interType = 0;
  uint32_t type = 0;
  uint32_t lhs = 0;
  uint32_t rhs = 0;
  uint32_t dest = 0;
  uint32_t gemmM = 0;
  uint32_t gemmK = 0;
  uint32_t gemmN = 0;
  uint32_t leftBatch = 0;
  uint32_t rightBatch = 0;
  uint32_t leftTrans = 0;
  uint32_t rightTrans = 0;
  DataFormat inputFormat = DataFormat::INT8;
  DataFormat outputFormat = DataFormat::INT8;
  bool psumEnabled = false;
  WaitPolicy waitPolicy = WaitPolicy::IssueOnly;
};

bool buildRdma1D(uint64_t ddrSrcAddr, uint32_t spmDstOffset, uint64_t bytes,
                 Dma1DDescriptor &descriptor, std::string *error);

bool buildWdma1D(uint32_t spmSrcOffset, uint64_t ddrDstAddr, uint64_t bytes,
                 Dma1DDescriptor &descriptor, std::string *error);

bool buildRdma(uint64_t ddrSrcAddr, uint32_t spmDstOffset, uint64_t byteCount,
               uint64_t innerBytes, std::array<int64_t, 3> strides,
               std::array<int64_t, 3> iterations, DmaDescriptor &descriptor,
               std::string *error);

bool buildWdma(uint32_t spmSrcOffset, uint64_t ddrDstAddr, uint64_t byteCount,
               uint64_t innerBytes, std::array<int64_t, 3> strides,
               std::array<int64_t, 3> iterations, DmaDescriptor &descriptor,
               std::string *error);

bool buildGatherScatter(uint32_t spmSrcOffset, uint32_t spmDstOffset,
                        uint64_t byteCount, uint64_t innerBytes,
                        std::array<int64_t, 3> srcStrides,
                        std::array<int64_t, 3> srcIterations,
                        std::array<int64_t, 3> dstStrides,
                        std::array<int64_t, 3> dstIterations,
                        GatherScatterDescriptor &descriptor,
                        std::string *error);

bool buildGemm(int64_t m, int64_t k, int64_t n, GemmDescriptor &descriptor,
               std::string *error);

bool buildRdmaRegisterPacket(const DmaDescriptor &descriptor, DataFormat format,
                             DmaRegisterPacket &packet, std::string *error);

bool buildWdmaRegisterPacket(const DmaDescriptor &descriptor, DataFormat format,
                             DmaRegisterPacket &packet, std::string *error);

bool buildGatherScatterRegisterPacket(const GatherScatterDescriptor &descriptor,
                                      GatherScatterRegisterPacket &packet,
                                      std::string *error);

bool buildGemmRegisterPacket(const GemmDescriptor &descriptor,
                             uint32_t lhsSpmOffset, uint32_t rhsSpmOffset,
                             uint32_t destSpmOffset, DataFormat inputFormat,
                             DataFormat outputFormat,
                             GemmRegisterPacket &packet, std::string *error);

} // namespace abi
} // namespace wafer

#endif // WAFER_ABI_TILEABI_H
