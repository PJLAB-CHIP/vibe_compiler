//===- TileAbi.h - Tile C ABI descriptor builders -----------------*- C++ -*-===//

#ifndef WAFER_ABI_TILEABI_H
#define WAFER_ABI_TILEABI_H

#include <cstdint>
#include <string>

namespace wafer {
namespace abi {

constexpr uint64_t kDdrLowerBound = 0x280000000ULL;
constexpr uint64_t kUsableSpmBytes = 0x2f0000ULL;

enum class WaitPolicy {
  IssueOnly,
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

bool buildRdma1D(uint64_t ddrSrcAddr, uint32_t spmDstOffset, uint64_t bytes,
                 Dma1DDescriptor &descriptor, std::string *error);

bool buildWdma1D(uint32_t spmSrcOffset, uint64_t ddrDstAddr, uint64_t bytes,
                 Dma1DDescriptor &descriptor, std::string *error);

bool buildGemm(int64_t m, int64_t k, int64_t n, GemmDescriptor &descriptor,
               std::string *error);

} // namespace abi
} // namespace wafer

#endif // WAFER_ABI_TILEABI_H
