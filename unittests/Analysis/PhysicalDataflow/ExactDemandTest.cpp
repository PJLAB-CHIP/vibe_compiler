//===- ExactDemandTest.cpp --------------------------------------------===//

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

#include "gtest/gtest.h"

namespace {

using namespace wafer::analysis;

TEST(ExactDemandOutcomeTest, ClassifiesEveryTypedOutcomeWithoutDiagnostics) {
  ExactDemandOutcome satisfied = ExactDemandProof{};
  ExactDemandOutcome unsupported = UnsupportedDemandSemantics{};
  ExactDemandOutcome exhausted = DemandWorkLimitReached{};
  ExactDemandOutcome invalid = InvalidSpatialAssignment{};
  ExactDemandOutcome broken = BrokenDemandContract{};

  EXPECT_EQ(classifyExactDemandOutcome(satisfied),
            ExactDemandOutcomeCategory::Satisfied);
  EXPECT_EQ(classifyExactDemandOutcome(unsupported),
            ExactDemandOutcomeCategory::UnsupportedSemantics);
  EXPECT_EQ(classifyExactDemandOutcome(exhausted),
            ExactDemandOutcomeCategory::IndeterminateResourceExhaustion);
  EXPECT_EQ(classifyExactDemandOutcome(invalid),
            ExactDemandOutcomeCategory::CompilerContractError);
  EXPECT_EQ(classifyExactDemandOutcome(broken),
            ExactDemandOutcomeCategory::CompilerContractError);
  EXPECT_NE(getExactDemandProof(satisfied), nullptr);
  EXPECT_EQ(getExactDemandProof(unsupported), nullptr);

  std::get<UnsupportedDemandSemantics>(unsupported).detail = "changed text";
  EXPECT_EQ(classifyExactDemandOutcome(unsupported),
            ExactDemandOutcomeCategory::UnsupportedSemantics);
}

TEST(ExactDemandOutcomeTest, ExactSetKeepsItsConstructionForm) {
  IndexSetResult set =
      IndexRelation::staticRectangularDomain({2, 3}, {4, 5});
  ASSERT_TRUE(set.isExact());
  StaticRectangularIndexSet box{{2, 3}, {4, 5}};
  ExactIndexSet exact(std::move(*set.set), ExactIndexSetForm::BoxUnion, {box});
  EXPECT_EQ(exact.getRank(), 2u);
  EXPECT_FALSE(exact.isEmpty());
  EXPECT_EQ(exact.getForm(), ExactIndexSetForm::BoxUnion);
  ASSERT_EQ(exact.getBoxes().size(), 1u);
  EXPECT_EQ(exact.getBoxes().front().offsets,
            (llvm::SmallVector<int64_t, 4>{2, 3}));
}

TEST(ExactDemandOutcomeTest,
     FiniteGeneralPresburgerRecoversAlignedAndRaggedBoxesExactly) {
  for (int64_t extent : {1024, 1025}) {
    IndexSetResult rectangular =
        IndexRelation::staticRectangularDomain({0, 7, 0}, {2, extent, 128});
    ASSERT_TRUE(rectangular.isExact());
    ExactIndexSet general(std::move(*rectangular.set),
                          ExactIndexSetForm::GeneralPresburger);
    mlir::FailureOr<ExactIndexSet> normalized =
        normalizeFiniteExactIndexSet(general);
    ASSERT_TRUE(mlir::succeeded(normalized));
    EXPECT_EQ(normalized->getForm(), ExactIndexSetForm::BoxUnion);
    ASSERT_EQ(normalized->getBoxes().size(), 1u);
    EXPECT_EQ(normalized->getBoxes().front().offsets,
              (llvm::SmallVector<int64_t, 4>{0, 7, 0}));
    EXPECT_EQ(normalized->getBoxes().front().sizes,
              (llvm::SmallVector<int64_t, 4>{2, extent, 128}));
    EXPECT_EQ(normalized->getPresburgerSet().isEqual(
                  general.getPresburgerSet()),
              true);
  }
}

} // namespace
