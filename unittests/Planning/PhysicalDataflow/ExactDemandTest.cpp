//===- ExactDemandTest.cpp --------------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/ExactDemand.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>

namespace {

using namespace wafer::analysis;

ExactIndexSet
makeBoxUnion(unsigned rank, llvm::ArrayRef<StaticRectangularIndexSet> boxes,
             ExactIndexSetForm form = ExactIndexSetForm::BoxUnion) {
  auto set = mlir::presburger::PresburgerSet::getEmpty(
      mlir::presburger::PresburgerSpace::getSetSpace(rank));
  for (const auto &box : boxes) {
    // Construct the oracle directly: unlike tensor coordinates, exact integer
    // sets may have negative origins and endpoints beyond query work limits.
    mlir::presburger::IntegerPolyhedron piece(
        mlir::presburger::PresburgerSpace::getSetSpace(rank));
    for (unsigned axis = 0; axis < rank; ++axis) {
      piece.addBound(mlir::presburger::BoundType::LB, axis, box.offsets[axis]);
      piece.addBound(mlir::presburger::BoundType::UB, axis,
                     box.offsets[axis] + box.sizes[axis] - 1);
    }
    set.unionInPlace(mlir::presburger::PresburgerSet(piece));
  }
  return ExactIndexSet(std::move(set), form, boxes);
}

bool boxesContain(llvm::ArrayRef<StaticRectangularIndexSet> boxes,
                  llvm::ArrayRef<int64_t> point) {
  return llvm::any_of(boxes, [&](const auto &box) {
    for (unsigned axis = 0; axis < point.size(); ++axis)
      if (point[axis] < box.offsets[axis] ||
          point[axis] >= box.offsets[axis] + box.sizes[axis])
        return false;
    return true;
  });
}

TEST(ExactDemandOutcomeTest, CoalescesRowsAcrossAxesAndRaggedExtentsExactly) {
  for (int64_t extent : {1024, 1025, 1031})
    for (unsigned axis : {0u, 1u, 2u}) {
      SCOPED_TRACE(std::to_string(extent) + ":" + std::to_string(axis));
      llvm::SmallVector<StaticRectangularIndexSet> boxes;
      for (int64_t row = 0; row < extent; ++row) {
        StaticRectangularIndexSet box{{7, 11, 13}, {2, 4, 8}};
        box.offsets[axis] += row;
        box.sizes[axis] = 1;
        boxes.push_back(box);
      }
      auto source = makeBoxUnion(3, boxes);
      auto normalized = normalizeFiniteExactIndexSet(source);
      ASSERT_TRUE(mlir::succeeded(normalized));
      ASSERT_EQ(normalized->getBoxes().size(), 1u);
      auto expected = boxes.front();
      expected.sizes[axis] = extent;
      EXPECT_EQ(normalized->getBoxes().front().offsets, expected.offsets);
      EXPECT_EQ(normalized->getBoxes().front().sizes, expected.sizes);
      auto rectangle = IndexRelation::staticRectangularDomain(expected.offsets,
                                                              expected.sizes);
      ASSERT_TRUE(rectangle.isExact());
      EXPECT_TRUE(source.getPresburgerSet().isEqual(*rectangle.set));
      auto again = normalizeFiniteExactIndexSet(*normalized);
      ASSERT_TRUE(mlir::succeeded(again));
      ASSERT_EQ(again->getBoxes().size(), 1u);
      EXPECT_EQ(again->getBoxes().front().sizes, expected.sizes);
      std::reverse(boxes.begin(), boxes.end());
      auto reversed = normalizeFiniteExactIndexSet(makeBoxUnion(3, boxes));
      ASSERT_TRUE(mlir::succeeded(reversed));
      ASSERT_EQ(reversed->getBoxes().size(), 1u);
      EXPECT_EQ(reversed->getBoxes().front().offsets, expected.offsets);
      EXPECT_EQ(reversed->getBoxes().front().sizes, expected.sizes);
    }
}

TEST(ExactDemandOutcomeTest, CoalescesMultiaxisGridAndGeneralPresburgerPieces) {
  for (int64_t extent : {1024, 1025, 1031})
    for (auto form :
         {ExactIndexSetForm::BoxUnion, ExactIndexSetForm::GeneralPresburger}) {
      llvm::SmallVector<StaticRectangularIndexSet> boxes;
      for (int64_t batch = 0; batch < 2; ++batch)
        // Eight disjuncts respect the existing general-query work limit.
        for (int64_t part = 0; part < 2; ++part)
          for (int64_t column = 0; column < 16; column += 8)
            boxes.push_back({{batch, part * (extent / 2), 0, column},
                             {1, extent / 2 + (part ? extent % 2 : 0), 1, 8}});
      auto normalized =
          normalizeFiniteExactIndexSet(makeBoxUnion(4, boxes, form));
      ASSERT_TRUE(mlir::succeeded(normalized));
      ASSERT_EQ(normalized->getBoxes().size(), 1u);
      EXPECT_EQ(normalized->getBoxes().front().offsets,
                (llvm::SmallVector<int64_t, 4>{0, 0, 0, 0}));
      EXPECT_EQ(normalized->getBoxes().front().sizes,
                (llvm::SmallVector<int64_t, 4>{2, extent, 1, 16}));
    }
}

TEST(ExactDemandOutcomeTest, PreservesHolesAndNonrectangularUnions) {
  // Tiny geometry permits a complete independent point oracle. Real-scale
  // row and multi-axis normalization is covered by the tests above.
  for (llvm::SmallVector<StaticRectangularIndexSet> boxes :
       {llvm::SmallVector<StaticRectangularIndexSet>{{{0, 0, 0}, {1, 4, 2}},
                                                     {{0, 4, 0}, {1, 2, 4}}},
        llvm::SmallVector<StaticRectangularIndexSet>{{{0, -2, 0}, {1, 2, 4}},
                                                     {{0, 1, 0}, {1, 2, 4}}},
        llvm::SmallVector<StaticRectangularIndexSet>{{{0, 0, 0}, {1, 4, 2}},
                                                     {{0, 0, 0}, {1, 4, 2}},
                                                     {{0, 2, 0}, {1, 4, 2}}}}) {
    auto normalized = normalizeFiniteExactIndexSet(makeBoxUnion(3, boxes));
    ASSERT_TRUE(mlir::succeeded(normalized));
    auto output = makeBoxUnion(3, normalized->getBoxes());
    EXPECT_TRUE(
        output.getPresburgerSet().isEqual(normalized->getPresburgerSet()));
    for (int64_t row = -3; row <= 6; ++row)
      for (int64_t column = -1; column <= 4; ++column)
        EXPECT_EQ(boxesContain(boxes, {0, row, column}),
                  boxesContain(normalized->getBoxes(), {0, row, column}));
  }
}

TEST(ExactDemandOutcomeTest, HandlesEmptyScalarAndRejectsMalformedBoxes) {
  auto empty = normalizeFiniteExactIndexSet(makeBoxUnion(3, {}));
  ASSERT_TRUE(mlir::succeeded(empty));
  EXPECT_TRUE(empty->getBoxes().empty());
  EXPECT_TRUE(empty->isEmpty());
  auto scalar =
      normalizeFiniteExactIndexSet(makeBoxUnion(0, {{{}, {}}, {{}, {}}}));
  ASSERT_TRUE(mlir::succeeded(scalar));
  EXPECT_EQ(scalar->getBoxes().size(), 1u);

  auto valid = makeBoxUnion(3, {{{0, 0, 0}, {1, 1024, 8}}});
  for (const auto &box :
       {StaticRectangularIndexSet{{0, 0}, {1, 1024}},
        StaticRectangularIndexSet{{0, 0, 0}, {1, 1024}},
        StaticRectangularIndexSet{{0, 0, 0}, {1, 0, 8}},
        StaticRectangularIndexSet{{0, std::numeric_limits<int64_t>::max(), 0},
                                  {1, 2, 8}}}) {
    ExactIndexSet malformed(valid.getPresburgerSet(),
                            ExactIndexSetForm::BoxUnion, {box});
    EXPECT_TRUE(mlir::failed(normalizeFiniteExactIndexSet(malformed)));
  }
  constexpr int64_t limit = std::numeric_limits<int64_t>::max();
  auto wide = makeBoxUnion(
      3, {{{0, -limit, 0}, {1, limit, 8}}, {{0, 0, 0}, {1, limit, 8}}});
  auto normalized = normalizeFiniteExactIndexSet(wide);
  ASSERT_TRUE(mlir::succeeded(normalized));
  EXPECT_EQ(normalized->getBoxes().size(), 2u);

  mlir::presburger::IntegerPolyhedron triangle(
      valid.getPresburgerSet().getAllDisjuncts().front());
  triangle.addInequality({0, 1, -1, 0});
  ExactIndexSet nonrectangular(mlir::presburger::PresburgerSet(triangle),
                               ExactIndexSetForm::GeneralPresburger);
  EXPECT_TRUE(mlir::failed(normalizeFiniteExactIndexSet(nonrectangular)));
}

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
