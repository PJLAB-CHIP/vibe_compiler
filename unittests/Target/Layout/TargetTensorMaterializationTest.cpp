//===- TargetTensorMaterializationTest.cpp - Static conversion contract --===//

#include "Wafer/Target/Layout/TargetTensorMaterialization.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <string>

namespace {

TEST(TargetTensorMaterializationTest, RequiresExplicitConversionParameter) {
  auto identity = wafer::TargetTensorMaterializationAction::create(
      wafer::LogicalFormat::F16, wafer::LogicalFormat::F16,
      /*parameter=*/std::nullopt);
  ASSERT_TRUE(static_cast<bool>(identity))
      << llvm::toString(identity.takeError());
  EXPECT_EQ(identity->getKind(),
            wafer::TargetTensorMaterializationKind::Identity);

  auto parameterizedIdentity = wafer::TargetTensorMaterializationAction::create(
      wafer::LogicalFormat::F16, wafer::LogicalFormat::F16,
      wafer::TargetConvertParameter::roundingMode(
          wafer::TargetRoundingMode::NearestEven));
  ASSERT_FALSE(static_cast<bool>(parameterizedIdentity));
  EXPECT_NE(llvm::toString(parameterizedIdentity.takeError())
                .find("identity target tensor materialization"),
            std::string::npos);

  auto missingRounding = wafer::TargetTensorMaterializationAction::create(
      wafer::LogicalFormat::F16, wafer::LogicalFormat::BF16,
      /*parameter=*/std::nullopt);
  ASSERT_FALSE(static_cast<bool>(missingRounding));
  EXPECT_NE(llvm::toString(missingRounding.takeError())
                .find("requires an explicit rounding mode"),
            std::string::npos);

  auto wrongParameter = wafer::TargetTensorMaterializationAction::create(
      wafer::LogicalFormat::F16, wafer::LogicalFormat::BF16,
      wafer::TargetConvertParameter::zeroPoint(0));
  ASSERT_FALSE(static_cast<bool>(wrongParameter));
  EXPECT_NE(llvm::toString(wrongParameter.takeError())
                .find("requires an explicit rounding mode"),
            std::string::npos);
}

TEST(TargetTensorMaterializationTest, AppliesSelectedRoundingMode) {
  auto action = wafer::TargetTensorMaterializationAction::create(
      wafer::LogicalFormat::F16, wafer::LogicalFormat::BF16,
      wafer::TargetConvertParameter::roundingMode(
          wafer::TargetRoundingMode::NearestEven));
  ASSERT_TRUE(static_cast<bool>(action)) << llvm::toString(action.takeError());
  EXPECT_EQ(action->getKind(),
            wafer::TargetTensorMaterializationKind::ValueConversion);
  ASSERT_TRUE(action->getParameter());
  EXPECT_EQ(action->getParameter()->getRoundingMode(),
            wafer::TargetRoundingMode::NearestEven);

  auto converted = wafer::materializeTargetTensorValue(
      *action, {wafer::LogicalFormat::F16, UINT64_C(0x3c00)});
  ASSERT_TRUE(static_cast<bool>(converted))
      << llvm::toString(converted.takeError());
  EXPECT_EQ(converted->format, wafer::LogicalFormat::BF16);
  EXPECT_EQ(converted->bits, UINT64_C(0x3f80));
}

} // namespace
