#include "Wafer/ABI/TileAbi.h"

#include "gtest/gtest.h"

namespace {

TEST(TileAbiTest, BuildsRdmaDescriptorWithByteRanges) {
  wafer::abi::Dma1DDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildRdma1D(
      /*ddrSrcAddr=*/0x280000000ULL, /*spmDstOffset=*/0,
      /*bytes=*/64, descriptor, &error))
      << error;

  EXPECT_EQ(descriptor.ddrAddress, 0x280000000ULL);
  EXPECT_EQ(descriptor.spmOffset, 0u);
  EXPECT_EQ(descriptor.bytes, 64u);
  EXPECT_EQ(descriptor.ddrEndExclusive, 0x280000040ULL);
  EXPECT_EQ(descriptor.spmEndExclusive, 64u);
  EXPECT_EQ(descriptor.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsWdmaDescriptorWithByteRanges) {
  wafer::abi::Dma1DDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildWdma1D(
      /*spmSrcOffset=*/2048, /*ddrDstAddr=*/0x280001000ULL,
      /*bytes=*/128, descriptor, &error))
      << error;

  EXPECT_EQ(descriptor.ddrAddress, 0x280001000ULL);
  EXPECT_EQ(descriptor.spmOffset, 2048u);
  EXPECT_EQ(descriptor.bytes, 128u);
  EXPECT_EQ(descriptor.ddrEndExclusive, 0x280001080ULL);
  EXPECT_EQ(descriptor.spmEndExclusive, 2176u);
  EXPECT_EQ(descriptor.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsGemmDescriptor) {
  wafer::abi::GemmDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(
      wafer::abi::buildGemm(/*m=*/4, /*k=*/8, /*n=*/16, descriptor, &error))
      << error;

  EXPECT_EQ(descriptor.m, 4);
  EXPECT_EQ(descriptor.k, 8);
  EXPECT_EQ(descriptor.n, 16);
  EXPECT_EQ(descriptor.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsStridedDmaDescriptor) {
  wafer::abi::DmaDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildRdma(
      /*ddrSrcAddr=*/0x280000200ULL, /*spmDstOffset=*/65536,
      /*byteCount=*/512, /*innerBytes=*/64,
      /*strides=*/{128, 0, 0}, /*iterations=*/{4, 1, 1}, descriptor, &error))
      << error;

  EXPECT_EQ(descriptor.ddrAddress, 0x280000200ULL);
  EXPECT_EQ(descriptor.spmOffset, 65536u);
  EXPECT_EQ(descriptor.byteCount, 512u);
  EXPECT_EQ(descriptor.innerBytes, 64u);
  EXPECT_EQ(descriptor.strides[0], 128);
  EXPECT_EQ(descriptor.iterations[0], 4);
  EXPECT_EQ(descriptor.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsRdmaRegisterPacketFromWrapperContract) {
  wafer::abi::DmaDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildRdma(
      /*ddrSrcAddr=*/0x280000200ULL, /*spmDstOffset=*/65536,
      /*byteCount=*/512, /*innerBytes=*/64,
      /*strides=*/{128, 0, 0}, /*iterations=*/{4, 1, 1}, descriptor, &error))
      << error;

  wafer::abi::DmaRegisterPacket packet;
  ASSERT_TRUE(wafer::abi::buildRdmaRegisterPacket(
      descriptor, wafer::abi::DataFormat::FP16, packet, &error))
      << error;

  EXPECT_EQ(packet.interType, wafer::abi::kInterTypeRDMA);
  EXPECT_EQ(packet.src, 0x280000200ULL);
  EXPECT_EQ(packet.dst, 65536ULL);
  EXPECT_EQ(packet.elemCount, 32u);
  EXPECT_EQ(packet.format, wafer::abi::DataFormat::FP16);
  EXPECT_EQ(packet.stride0, 128u);
  EXPECT_EQ(packet.iteration0, 3u);
  EXPECT_EQ(packet.stride1, 0u);
  EXPECT_EQ(packet.iteration1, 0u);
  EXPECT_EQ(packet.stride2, 0u);
  EXPECT_EQ(packet.iteration2, 0u);
  EXPECT_EQ(packet.srcEnd, 0x2800003bfULL);
  EXPECT_EQ(packet.dstEnd, 66047ULL);
  EXPECT_EQ(packet.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsWdmaRegisterPacketFromWrapperContract) {
  wafer::abi::DmaDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildWdma(
      /*spmSrcOffset=*/2048, /*ddrDstAddr=*/0x280001000ULL,
      /*byteCount=*/128, /*innerBytes=*/32,
      /*strides=*/{64, 0, 0}, /*iterations=*/{4, 1, 1}, descriptor, &error))
      << error;

  wafer::abi::DmaRegisterPacket packet;
  ASSERT_TRUE(wafer::abi::buildWdmaRegisterPacket(
      descriptor, wafer::abi::DataFormat::FP16, packet, &error))
      << error;

  EXPECT_EQ(packet.interType, wafer::abi::kInterTypeWDMA);
  EXPECT_EQ(packet.src, 2048ULL);
  EXPECT_EQ(packet.dst, 0x280001000ULL);
  EXPECT_EQ(packet.elemCount, 16u);
  EXPECT_EQ(packet.format, wafer::abi::DataFormat::FP16);
  EXPECT_EQ(packet.iteration0, 3u);
  EXPECT_EQ(packet.srcEnd, 2175ULL);
  EXPECT_EQ(packet.dstEnd, 0x2800010dfULL);
  EXPECT_EQ(packet.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsGatherScatterRegisterPacketWithByteSize) {
  wafer::abi::GatherScatterDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildGatherScatter(
      /*spmSrcOffset=*/4096, /*spmDstOffset=*/8192,
      /*byteCount=*/128, /*innerBytes=*/32,
      /*srcStrides=*/{64, 0, 0}, /*srcIterations=*/{4, 1, 1},
      /*dstStrides=*/{96, 0, 0}, /*dstIterations=*/{4, 1, 1}, descriptor,
      &error))
      << error;

  wafer::abi::GatherScatterRegisterPacket packet;
  ASSERT_TRUE(
      wafer::abi::buildGatherScatterRegisterPacket(descriptor, packet, &error))
      << error;

  EXPECT_EQ(packet.interType, wafer::abi::kInterTypeTDMA);
  EXPECT_EQ(packet.opcode, wafer::abi::kTdmaGatherScatterOpcode);
  EXPECT_EQ(packet.src0, 4096u);
  EXPECT_EQ(packet.dst, 8192u);
  EXPECT_EQ(packet.elemCount, 32u);
  EXPECT_EQ(packet.srcStride0, 64u);
  EXPECT_EQ(packet.srcIteration0, 3u);
  EXPECT_EQ(packet.dstStride0, 96u);
  EXPECT_EQ(packet.dstIteration0, 3u);
  EXPECT_EQ(packet.src0End, 4319u);
  EXPECT_EQ(packet.dstEnd, 8511u);
  EXPECT_EQ(packet.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, BuildsGemmRegisterPacketFromWrapperContract) {
  wafer::abi::GemmDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(
      wafer::abi::buildGemm(/*m=*/4, /*k=*/8, /*n=*/16, descriptor, &error))
      << error;

  wafer::abi::GemmRegisterPacket packet;
  ASSERT_TRUE(wafer::abi::buildGemmRegisterPacket(
      descriptor, /*lhsSpmOffset=*/65536, /*rhsSpmOffset=*/65792,
      /*destSpmOffset=*/66048, wafer::abi::DataFormat::FP16,
      wafer::abi::DataFormat::FP16, packet, &error))
      << error;

  EXPECT_EQ(packet.interType, wafer::abi::kInterTypeNEUR);
  EXPECT_EQ(packet.type, wafer::abi::kNeTypeGemm);
  EXPECT_EQ(packet.lhs, 65536u);
  EXPECT_EQ(packet.rhs, 65792u);
  EXPECT_EQ(packet.dest, 66048u);
  EXPECT_EQ(packet.gemmM, 4u);
  EXPECT_EQ(packet.gemmK, 8u);
  EXPECT_EQ(packet.gemmN, 16u);
  EXPECT_EQ(packet.leftBatch, 1u);
  EXPECT_EQ(packet.rightBatch, 1u);
  EXPECT_EQ(packet.leftTrans, 0u);
  EXPECT_EQ(packet.rightTrans, 0u);
  EXPECT_EQ(packet.inputFormat, wafer::abi::DataFormat::FP16);
  EXPECT_EQ(packet.outputFormat, wafer::abi::DataFormat::FP16);
  EXPECT_FALSE(packet.psumEnabled);
  EXPECT_EQ(packet.waitPolicy, wafer::abi::WaitPolicy::IssueOnly);
}

TEST(TileAbiTest, RejectsDmaRegisterPacketWithUnalignedElementCount) {
  wafer::abi::DmaDescriptor descriptor;
  std::string error;

  ASSERT_TRUE(wafer::abi::buildRdma(
      /*ddrSrcAddr=*/0x280000000ULL, /*spmDstOffset=*/0,
      /*byteCount=*/9, /*innerBytes=*/3,
      /*strides=*/{3, 0, 0}, /*iterations=*/{3, 1, 1}, descriptor, &error))
      << error;

  wafer::abi::DmaRegisterPacket packet;
  EXPECT_FALSE(wafer::abi::buildRdmaRegisterPacket(
      descriptor, wafer::abi::DataFormat::FP16, packet, &error));
  EXPECT_NE(error.find("DMA inner byte count must be divisible by format "
                       "element bytes"),
            std::string::npos);
}

TEST(TileAbiTest, RejectsInvalidStridedDmaDescriptor) {
  wafer::abi::DmaDescriptor descriptor;
  std::string error;

  EXPECT_FALSE(wafer::abi::buildRdma(
      /*ddrSrcAddr=*/0x280000000ULL, /*spmDstOffset=*/0,
      /*byteCount=*/64, /*innerBytes=*/0,
      /*strides=*/{16, 0, 0}, /*iterations=*/{1, 1, 1}, descriptor, &error));
  EXPECT_NE(error.find("DMA inner byte count must be positive"),
            std::string::npos);

  error.clear();
  EXPECT_FALSE(wafer::abi::buildRdma(
      /*ddrSrcAddr=*/0x280000000ULL, /*spmDstOffset=*/0,
      /*byteCount=*/64, /*innerBytes=*/128,
      /*strides=*/{16, 0, 0}, /*iterations=*/{1, 1, 1}, descriptor, &error));
  EXPECT_NE(error.find("DMA inner byte count must not exceed byte count"),
            std::string::npos);
}

TEST(TileAbiTest, RejectsInvalidDmaRanges) {
  wafer::abi::Dma1DDescriptor descriptor;
  std::string error;

  EXPECT_FALSE(wafer::abi::buildRdma1D(
      /*ddrSrcAddr=*/0x27fffffffULL, /*spmDstOffset=*/0,
      /*bytes=*/64, descriptor, &error));
  EXPECT_NE(error.find("DDR address below lower bound"), std::string::npos);

  error.clear();
  EXPECT_FALSE(wafer::abi::buildWdma1D(
      /*spmSrcOffset=*/0x2efff0, /*ddrDstAddr=*/0x280000000ULL,
      /*bytes=*/32, descriptor, &error));
  EXPECT_NE(error.find("SPM range exceeds usable capacity"), std::string::npos);
}

} // namespace
