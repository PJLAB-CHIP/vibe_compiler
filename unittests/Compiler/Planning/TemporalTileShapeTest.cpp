//===- TemporalTileShapeTest.cpp -------------------------------------===//

#include "../TestSupport/CardExecutableTestSupport.h"

#include "gtest/gtest.h"

namespace {
using namespace wafer::compiler::testing;

TEST(TemporalTileShapeTest,
     DerivationMakesStrictProgressAcrossAlignedClasses) {
  const wafer::TargetMemoryPolicy memory =
      wafer::getDefaultWaferTargetPolicy().memory;

  // This is the conv-mixed-DAG reduction boundary: 16-way spatial mapping
  // leaves a 1x2 output tile, while the component's large convolution inputs
  // make its aligned tensor-byte scale exceed SPM at that extent. 16 -> 17
  // waves still maps extent 32 back to tile size 2, so the old waves+1 update
  // did not progress. The adjacent distinct class is tile size 1.
  EXPECT_EQ(wafer::compiler::detail::deriveCapacityTemporalTileShape(
                {1, 2}, /*elementBytes=*/2,
                /*tensorMultiplicity=*/16384, memory,
                /*additionalWaveRefinements=*/0),
            (llvm::SmallVector<int64_t, 4>{1, 1}));

  // Overflow in bytes-per-element-set must saturate before division; wrapping
  // would incorrectly enlarge the capacity budget and retain extent 2.
  EXPECT_EQ(wafer::compiler::detail::deriveCapacityTemporalTileShape(
                {1, 2}, std::numeric_limits<uint64_t>::max(),
                /*tensorMultiplicity=*/2, memory,
                /*additionalWaveRefinements=*/0),
            (llvm::SmallVector<int64_t, 4>{1, 1}));
}


} // namespace
