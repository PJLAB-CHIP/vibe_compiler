//===- SpatialPlacementTest.cpp -----------------------------------------===//

#include "Wafer/Planning/Search/SpatialPlacement.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>

namespace {

using wafer::TileId;
using wafer::compiler::detail::CardSpatialPlacementAssignment;
using wafer::compiler::detail::CardSpatialPlacementDomain;
using wafer::compiler::detail::SpatialPlacementAssignment;
using wafer::compiler::detail::SpatialPlacementDomain;
using wafer::compiler::detail::StructuredDAGAnalysis;
using wafer::compiler::detail::StructuredDAGExactDemandQuery;

class SpatialPlacementTest : public ::testing::Test {
protected:
  SpatialPlacementTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parseMatmul() {
    return mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  func.func @matmul(%lhs: tensor<3x2xf16>, %rhs: tensor<2x2xf16>)
      -> tensor<3x2xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<3x2xf16>
    %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<3x2xf16>)
        -> tensor<3x2xf16>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<3x2xf16>, tensor<2x2xf16>)
        outs(%init : tensor<3x2xf16>) -> tensor<3x2xf16>
    %mapped_empty = tensor.empty() : tensor<3x2xf16>
    %mapped = linalg.map ins(%result : tensor<3x2xf16>)
        outs(%mapped_empty : tensor<3x2xf16>) (%value: f16) {
      %sum = arith.addf %value, %value : f16
      linalg.yield %sum : f16
    }
    return %mapped : tensor<3x2xf16>
  }

}
)mlir",
        mlir::ParserConfig(context.get()));
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

static void enumerateTileSequences(
    llvm::ArrayRef<TileId> available, size_t count,
    llvm::SmallVectorImpl<TileId> &current,
    llvm::SmallVectorImpl<llvm::SmallVector<TileId, 4>> &result) {
  if (current.size() == count) {
    result.emplace_back(current.begin(), current.end());
    return;
  }
  for (TileId tile : available) {
    if (llvm::is_contained(current, tile))
      continue;
    current.push_back(tile);
    enumerateTileSequences(available, count, current, result);
    current.pop_back();
  }
}

static void
enumerateReference(wafer::compiler::detail::StructuredDAGNodeID node,
                   llvm::ArrayRef<int64_t> extents,
                   llvm::ArrayRef<uint8_t> reductions,
                   llvm::ArrayRef<TileId> available, size_t dimension,
                   llvm::SmallVectorImpl<uint32_t> &factors,
                   std::set<SpatialPlacementAssignment> &result) {
  if (dimension != extents.size()) {
    for (uint32_t factor = 1;
         factor <= static_cast<uint64_t>(extents[dimension]); ++factor) {
      factors.push_back(factor);
      enumerateReference(node, extents, reductions, available, dimension + 1,
                         factors, result);
      factors.pop_back();
    }
    return;
  }
  uint64_t participants = 1;
  for (uint32_t factor : factors)
    participants *= factor;
  if (participants > available.size())
    return;
  llvm::SmallVector<llvm::SmallVector<TileId, 4>, 16> embeddings;
  llvm::SmallVector<TileId, 4> current;
  enumerateTileSequences(available, participants, current, embeddings);
  for (llvm::ArrayRef<TileId> embedding : embeddings) {
    const bool needsMerge =
        llvm::any_of(llvm::zip_equal(factors, reductions), [](auto values) {
          return std::get<0>(values) > 1 && std::get<1>(values) != 0;
        });
    for (std::optional<TileId> merge :
         needsMerge
             ? llvm::to_vector<4>(llvm::map_range(
                   available,
                   [](TileId tile) { return std::optional<TileId>(tile); }))
             : llvm::SmallVector<std::optional<TileId>, 4>{std::nullopt}) {
      SpatialPlacementAssignment assignment;
      assignment.node = node;
      assignment.iteratorFactors.assign(factors.begin(), factors.end());
      assignment.tiles.assign(embedding.begin(), embedding.end());
      assignment.reductionMergeTile = merge;
      result.insert(std::move(assignment));
    }
  }
}

static void enumerateCardReference(
    llvm::ArrayRef<std::vector<SpatialPlacementAssignment>> nodeAssignments,
    size_t node, CardSpatialPlacementAssignment &current,
    std::set<CardSpatialPlacementAssignment> &result) {
  if (node == nodeAssignments.size()) {
    result.insert(current);
    return;
  }
  for (const SpatialPlacementAssignment &assignment : nodeAssignments[node]) {
    current.nodes.push_back(assignment);
    enumerateCardReference(nodeAssignments, node + 1, current, result);
    current.nodes.pop_back();
  }
}

static std::set<CardSpatialPlacementAssignment>
getCardReference(const StructuredDAGAnalysis &dag,
                 llvm::ArrayRef<TileId> available) {
  llvm::SmallVector<std::vector<SpatialPlacementAssignment>, 8> nodeAssignments;
  for (const auto &node : dag.getNodes()) {
    auto domain = SpatialPlacementDomain::create(node, available);
    EXPECT_TRUE(mlir::succeeded(domain));
    if (mlir::failed(domain))
      return {};
    std::vector<SpatialPlacementAssignment> assignments;
    std::optional<SpatialPlacementAssignment> current =
        domain->getFirstAssignment();
    while (current) {
      assignments.push_back(*current);
      auto next = domain->getNextAssignment(*current);
      EXPECT_TRUE(mlir::succeeded(next));
      if (mlir::failed(next))
        return {};
      current = *next;
    }
    nodeAssignments.push_back(std::move(assignments));
  }
  std::set<CardSpatialPlacementAssignment> result;
  CardSpatialPlacementAssignment current;
  enumerateCardReference(nodeAssignments, 0, current, result);
  return result;
}

TEST_F(SpatialPlacementTest,
       DomainMatchesIndependentMultiAxisPartitionAndEmbeddingEnumerator) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseMatmul();
  ASSERT_TRUE(module);
  mlir::func::FuncOp function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  ASSERT_EQ(dag->getNodes().size(), 3u);
  const auto &matmul = dag->getNodes()[1];
  llvm::SmallVector<TileId, 3> available{TileId(5), TileId(0), TileId(2)};
  auto domain = SpatialPlacementDomain::create(matmul, available);
  ASSERT_TRUE(mlir::succeeded(domain));
  EXPECT_EQ(domain->getIteratorExtents(), (llvm::ArrayRef<int64_t>{3, 2, 2}));
  EXPECT_EQ(domain->getReductionIterators(),
            (llvm::ArrayRef<uint8_t>{0, 0, 1}));

  std::vector<SpatialPlacementAssignment> actual;
  std::optional<SpatialPlacementAssignment> current =
      domain->getFirstAssignment();
  while (current) {
    actual.push_back(*current);
    auto next = domain->getNextAssignment(*current);
    ASSERT_TRUE(mlir::succeeded(next));
    current = *next;
  }
  EXPECT_TRUE(std::is_sorted(actual.begin(), actual.end()));
  EXPECT_EQ(actual.size(), 39u);

  std::set<SpatialPlacementAssignment> reference;
  llvm::SmallVector<uint32_t, 4> factors;
  enumerateReference(matmul.id, domain->getIteratorExtents(),
                     domain->getReductionIterators(),
                     domain->getAvailableTiles(), 0, factors, reference);
  EXPECT_EQ(std::set<SpatialPlacementAssignment>(actual.begin(), actual.end()),
            reference);

  SpatialPlacementAssignment reduction{
      matmul.id, {1, 1, 2}, {TileId(0), TileId(5)}, TileId(2)};
  EXPECT_TRUE(domain->contains(reduction));
  SpatialPlacementAssignment remainder{
      matmul.id, {2, 1, 1}, {TileId(5), TileId(0)}};
  EXPECT_TRUE(domain->contains(remainder));
}

TEST_F(SpatialPlacementTest, RejectsDuplicateOrOutOfDomainPhysicalEmbeddings) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseMatmul();
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto domain = SpatialPlacementDomain::create(
      dag->getNodes()[1], {TileId(0), TileId(2), TileId(5)});
  ASSERT_TRUE(mlir::succeeded(domain));

  EXPECT_FALSE(domain->contains(
      {dag->getNodes()[1].id, {1, 1, 2}, {TileId(0), TileId(0)}}));
  EXPECT_FALSE(domain->contains(
      {dag->getNodes()[1].id, {1, 1, 2}, {TileId(0), TileId(7)}}));
  EXPECT_FALSE(domain->contains(
      {dag->getNodes()[1].id, {1, 2}, {TileId(0), TileId(2)}}));
}

TEST_F(SpatialPlacementTest,
       AdmitsMultiAxisRemainderAndParallelReductionEmbeddings) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseMatmul();
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto domain = SpatialPlacementDomain::create(
      dag->getNodes()[1], {TileId(0), TileId(2), TileId(5), TileId(7)});
  ASSERT_TRUE(mlir::succeeded(domain));

  EXPECT_TRUE(domain->contains({dag->getNodes()[1].id,
                                {2, 2, 1},
                                {TileId(7), TileId(0), TileId(5), TileId(2)},
                                std::nullopt}));
  EXPECT_TRUE(domain->contains({dag->getNodes()[1].id,
                                {1, 2, 2},
                                {TileId(0), TileId(7), TileId(2), TileId(5)},
                                TileId(7)}));
}

TEST_F(SpatialPlacementTest,
       UnsupportedReductionCombinerKeepsOnlyUnitReductionFactor) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @product(%input: tensor<2xf16>, %init: tensor<f16>)
      -> tensor<f16> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> ()>],
        iterator_types = ["reduction"]
      } ins(%input : tensor<2xf16>) outs(%init : tensor<f16>) {
      ^bb0(%value: f16, %acc: f16):
        %product = arith.mulf %value, %acc : f16
        linalg.yield %product : f16
    } -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  ASSERT_EQ(dag->getNodes().size(), 1u);
  auto domain = SpatialPlacementDomain::create(dag->getNodes().front(),
                                               {TileId(0), TileId(2)});
  ASSERT_TRUE(mlir::succeeded(domain));
  EXPECT_EQ(domain->getReductionIterators(), (llvm::ArrayRef<uint8_t>{1}));
  EXPECT_EQ(domain->getMaximumFactors(), (llvm::ArrayRef<int64_t>{1}));
  EXPECT_FALSE(domain->contains(
      {dag->getNodes().front().id, {2}, {TileId(0), TileId(2)}}));
  auto first = domain->getFirstAssignment();
  auto second = domain->getNextAssignment(first);
  ASSERT_TRUE(mlir::succeeded(second));
  ASSERT_TRUE(*second);
  auto end = domain->getNextAssignment(**second);
  ASSERT_TRUE(mlir::succeeded(end));
  EXPECT_FALSE(*end);
}

TEST_F(SpatialPlacementTest,
       ScalarStructuredNodeHasEmptyFactorVectorAndEverySingletonTile) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parse(R"mlir(
module {
  func.func @scalar(%value: f16) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.fill ins(%value : f16) outs(%empty : tensor<f16>)
        -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  ASSERT_EQ(dag->getNodes().size(), 1u);
  auto domain = SpatialPlacementDomain::create(dag->getNodes().front(),
                                               {TileId(0), TileId(2)});
  ASSERT_TRUE(mlir::succeeded(domain));
  EXPECT_TRUE(domain->getIteratorExtents().empty());
  SpatialPlacementAssignment first = domain->getFirstAssignment();
  EXPECT_TRUE(first.iteratorFactors.empty());
  EXPECT_EQ(first.tiles, (llvm::SmallVector<TileId, 1>{TileId(0)}));
  auto second = domain->getNextAssignment(first);
  ASSERT_TRUE(mlir::succeeded(second));
  ASSERT_TRUE(*second);
  EXPECT_EQ((*second)->tiles, (llvm::SmallVector<TileId, 1>{TileId(2)}));
  auto end = domain->getNextAssignment(**second);
  ASSERT_TRUE(mlir::succeeded(end));
  EXPECT_FALSE(*end);
}

TEST_F(SpatialPlacementTest,
       CardDomainMatchesCartesianReferenceForChainBranchAndDiamond) {
  constexpr llvm::StringLiteral chain = R"mlir(
module {
  func.func @chain(%input: tensor<2xf16>) -> tensor<2xf16> {
    %empty0 = tensor.empty() : tensor<2xf16>
    %first = linalg.map ins(%input : tensor<2xf16>)
        outs(%empty0 : tensor<2xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %empty1 = tensor.empty() : tensor<2xf16>
    %second = linalg.map ins(%first : tensor<2xf16>)
        outs(%empty1 : tensor<2xf16>) (%value: f16) {
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    }
    return %second : tensor<2xf16>
  }
}
)mlir";
  constexpr llvm::StringLiteral branches = R"mlir(
module {
  func.func @branches(%lhs: tensor<2xf16>, %rhs: tensor<2xf16>)
      -> (tensor<2xf16>, tensor<2xf16>) {
    %empty0 = tensor.empty() : tensor<2xf16>
    %left = linalg.map ins(%lhs : tensor<2xf16>)
        outs(%empty0 : tensor<2xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %empty1 = tensor.empty() : tensor<2xf16>
    %right = linalg.map ins(%rhs : tensor<2xf16>)
        outs(%empty1 : tensor<2xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %left, %right : tensor<2xf16>, tensor<2xf16>
  }
}
)mlir";
  constexpr llvm::StringLiteral diamond = R"mlir(
module {
  func.func @diamond(%input: tensor<2xf16>) -> tensor<2xf16> {
    %empty0 = tensor.empty() : tensor<2xf16>
    %source = linalg.map ins(%input : tensor<2xf16>)
        outs(%empty0 : tensor<2xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    %empty1 = tensor.empty() : tensor<2xf16>
    %left = linalg.map ins(%source : tensor<2xf16>)
        outs(%empty1 : tensor<2xf16>) (%value: f16) {
      %next = arith.addf %value, %value : f16
      linalg.yield %next : f16
    }
    %empty2 = tensor.empty() : tensor<2xf16>
    %right = linalg.map ins(%source : tensor<2xf16>)
        outs(%empty2 : tensor<2xf16>) (%value: f16) {
      %next = arith.mulf %value, %value : f16
      linalg.yield %next : f16
    }
    %empty3 = tensor.empty() : tensor<2xf16>
    %join = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%left, %right : tensor<2xf16>, tensor<2xf16>)
        outs(%empty3 : tensor<2xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %old: f16):
        %sum = arith.addf %lhs, %rhs : f16
        linalg.yield %sum : f16
    } -> tensor<2xf16>
    return %join : tensor<2xf16>
  }
}
)mlir";

  for (llvm::StringRef source : {chain, branches, diamond}) {
    mlir::OwningOpRef<mlir::ModuleOp> module = parse(source);
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function, &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    llvm::SmallVector<TileId, 2> available{TileId(0), TileId(2)};
    auto domain = CardSpatialPlacementDomain::create(*dag, available);
    ASSERT_TRUE(mlir::succeeded(domain));

    std::vector<CardSpatialPlacementAssignment> actual;
    std::optional<CardSpatialPlacementAssignment> current =
        domain->getFirstAssignment();
    while (current) {
      actual.push_back(*current);
      auto evaluation =
          domain->evaluate(*dag, wafer::analysis::IREpoch::mint(), *current);
      EXPECT_EQ(evaluation.status,
                wafer::analysis::ExactDemandStatus::Satisfied)
          << evaluation.detail;
      auto next = domain->getNextAssignment(*current);
      ASSERT_TRUE(mlir::succeeded(next));
      current = *next;
    }
    EXPECT_TRUE(std::is_sorted(actual.begin(), actual.end()));
    EXPECT_EQ(
        std::set<CardSpatialPlacementAssignment>(actual.begin(), actual.end()),
        getCardReference(*dag, available));
  }
}

TEST_F(SpatialPlacementTest,
       ReductionPartitionProducesExactPartialOwnersAndMergeObligation) {
  mlir::OwningOpRef<mlir::ModuleOp> module = parseMatmul();
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  ASSERT_EQ(dag->getNodes().size(), 3u);

  wafer::analysis::IREpoch epoch = wafer::analysis::IREpoch::mint();
  auto cardDomain = CardSpatialPlacementDomain::create(
      *dag, {TileId(0), TileId(2), TileId(5)});
  ASSERT_TRUE(mlir::succeeded(cardDomain));
  CardSpatialPlacementAssignment assignment{{
      {dag->getNodes()[0].id, {1, 1}, {TileId(0)}},
      {dag->getNodes()[1].id, {1, 1, 2}, {TileId(0), TileId(5)}, TileId(2)},
      {dag->getNodes()[2].id, {1, 1}, {TileId(2)}},
  }};
  auto evaluation = cardDomain->evaluate(*dag, epoch, assignment);
  ASSERT_EQ(evaluation.status, wafer::analysis::ExactDemandStatus::Satisfied)
      << evaluation.detail;
  ASSERT_TRUE(evaluation.trial);

  wafer::analysis::LogicalShardTrial trial = std::move(*evaluation.trial);
  const wafer::analysis::LogicalNodeTrial &partial = trial.nodes[1];
  ASSERT_EQ(partial.executionShards.size(), 2u);
  EXPECT_TRUE(partial.executionShards[0].executionDomain->containsPoint(
      llvm::SmallVector<int64_t, 3>{0, 0, 0}));
  EXPECT_FALSE(partial.executionShards[0].executionDomain->containsPoint(
      llvm::SmallVector<int64_t, 3>{0, 0, 1}));
  EXPECT_TRUE(partial.executionShards[1].executionDomain->containsPoint(
      llvm::SmallVector<int64_t, 3>{0, 0, 1}));
  ASSERT_EQ(partial.bindings.size(), 2u);
  EXPECT_EQ(partial.reductionMergeTile, TileId(2));
  EXPECT_EQ(partial.bindings[0].role,
            wafer::analysis::TileRole::PartialReductionContribution);
  EXPECT_EQ(partial.bindings[1].role,
            wafer::analysis::TileRole::PartialReductionContribution);

  auto edge = llvm::find_if(dag->getEdges(), [&](const auto &candidate) {
    return candidate.producer == dag->getNodes()[1].id &&
           candidate.consumer == dag->getNodes()[2].id;
  });
  ASSERT_NE(edge, dag->getEdges().end());
  StructuredDAGExactDemandQuery query(*dag, epoch);
  wafer::analysis::ExactDemandResult demand = query.query(edge->id, trial);
  EXPECT_EQ(demand.status, wafer::analysis::ExactDemandStatus::Satisfied)
      << demand.detail;
  EXPECT_EQ(demand.role,
            wafer::analysis::TileRole::PartialReductionContribution);
  EXPECT_TRUE(demand.mergeObligation);
  EXPECT_EQ(demand.reductionMergeTile, TileId(2));
  EXPECT_EQ(demand.ownershipIntersections.size(), 2u);
}

} // namespace
