//===- TargetNumericCapabilityTest.cpp - Numeric admission tests --------===//

#include "Wafer/Target/TargetNumericCapability.h"

#include "gtest/gtest.h"

#include <set>
#include <tuple>

namespace {

using wafer::InstrElementwiseKind;
using wafer::LogicalFormat;
using wafer::TargetNumericUnsupportedReason;

constexpr wafer::TargetProfileId kProfile =
    wafer::TargetProfileId::waferTx81SingleCardKernelV1();
constexpr wafer::TargetNumericSemanticRequirement kSourceExact =
    wafer::TargetNumericSemanticRequirement::SourceExact;

TEST(TargetNumericCapabilityTest,
     ClosesElementwiseKindFormatAndExactSemanticRequirement) {
  llvm::ArrayRef<wafer::TargetCTElementwiseNumericCapabilityRecord> records =
      wafer::getTargetCTElementwiseNumericCapabilityRecords();
  ASSERT_EQ(records.size(), 128u);

  size_t supported = 0;
  size_t integerUnproven = 0;
  size_t parameterUnproven = 0;
  size_t signedZeroUnproven = 0;
  std::set<std::tuple<unsigned, unsigned, unsigned>> keys;
  for (const auto &record : records) {
    EXPECT_EQ(record.profile, kProfile);
    EXPECT_EQ(record.requirement, kSourceExact);
    EXPECT_TRUE(keys.insert({static_cast<unsigned>(record.kind),
                             static_cast<unsigned>(record.inputFormat),
                             static_cast<unsigned>(record.requirement)})
                    .second);
    if (record.isSupported()) {
      ++supported;
      EXPECT_EQ(record.unsupportedReason, TargetNumericUnsupportedReason::None);
      continue;
    }
    switch (record.unsupportedReason) {
    case TargetNumericUnsupportedReason::IntegerElementwisePolicyUnproven:
      ++integerUnproven;
      break;
    case TargetNumericUnsupportedReason::ElementwiseParameterPolicyUnproven:
      ++parameterUnproven;
      break;
    case TargetNumericUnsupportedReason::ExactSignedZeroPolicyUnproven:
      ++signedZeroUnproven;
      break;
    case TargetNumericUnsupportedReason::None:
      ADD_FAILURE() << "unsupported row has no reason";
      break;
    }
  }
  EXPECT_EQ(supported, 85u);
  EXPECT_EQ(integerUnproven, 31u);
  EXPECT_EQ(parameterUnproven, 9u);
  EXPECT_EQ(signedZeroUnproven, 3u);
}

TEST(TargetNumericCapabilityTest,
     RejectsUnqualifiedComputeWithoutDisablingQualifiedCompute) {
  const auto *f16Add = wafer::findTargetCTElementwiseNumericCapability(
      kProfile, InstrElementwiseKind::Add, LogicalFormat::F16, kSourceExact);
  const auto *i8Add = wafer::findTargetCTElementwiseNumericCapability(
      kProfile, InstrElementwiseKind::Add, LogicalFormat::I8, kSourceExact);
  const auto *i8Mul = wafer::findTargetCTElementwiseNumericCapability(
      kProfile, InstrElementwiseKind::Mul, LogicalFormat::I8, kSourceExact);
  const auto *f32Neg = wafer::findTargetCTElementwiseNumericCapability(
      kProfile, InstrElementwiseKind::Neg, LogicalFormat::F32, kSourceExact);

  ASSERT_NE(f16Add, nullptr);
  EXPECT_TRUE(f16Add->isSupported());
  ASSERT_NE(i8Add, nullptr);
  EXPECT_FALSE(i8Add->isSupported());
  EXPECT_EQ(i8Add->unsupportedReason,
            TargetNumericUnsupportedReason::IntegerElementwisePolicyUnproven);
  ASSERT_NE(i8Mul, nullptr);
  EXPECT_FALSE(i8Mul->isSupported());
  ASSERT_NE(f32Neg, nullptr);
  EXPECT_FALSE(f32Neg->isSupported());
  EXPECT_EQ(f32Neg->unsupportedReason,
            TargetNumericUnsupportedReason::ExactSignedZeroPolicyUnproven);
}

TEST(TargetNumericCapabilityTest,
     VersionedABIProfileInheritsNumericCompatibilityIdentity) {
  constexpr wafer::TargetProfileId kVersionedProfile =
      wafer::TargetProfileId::waferTx81SingleCardKernelV2();
  for (const auto &record :
       wafer::getTargetCTElementwiseNumericCapabilityRecords())
    EXPECT_EQ(wafer::findTargetCTElementwiseNumericCapability(
                  kVersionedProfile, record.kind, record.inputFormat,
                  record.requirement),
              &record);
}

} // namespace
