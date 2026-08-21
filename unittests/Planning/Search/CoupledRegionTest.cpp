//===- CoupledRegionTest.cpp ------------------------------------------===//

#include "Wafer/Planning/Search/CoupledRegion.h"

#include "Wafer/Planning/PhysicalDataflow/StructuredDAGPlacement.h"
#include "TestSupport/Planning/SpatialDemandTestSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>

namespace {

using wafer::TileId;
using wafer::compiler::detail::CoupledRegionAssignment;
using wafer::compiler::detail::CoupledRegionDomain;
using wafer::compiler::detail::CoupledRegionGroup;
using wafer::compiler::detail::StructuredDAGAnalysis;
using wafer::compiler::detail::StructuredDAGNodePlacement;

class CoupledRegionTest : public ::testing::Test {
protected:
  CoupledRegionTest() {
    registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
    mlir::linalg::registerTilingInterfaceExternalModels(registry);
    context = std::make_unique<mlir::MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(
        source, mlir::ParserConfig(context.get()));
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

bool isConnectedGroup(const StructuredDAGAnalysis &dag,
                      llvm::ArrayRef<uint32_t> labels, uint32_t group) {
  llvm::SmallVector<size_t, 8> members;
  for (auto [index, label] : llvm::enumerate(labels))
    if (label == group)
      members.push_back(index);
  if (members.size() < 2)
    return !members.empty();
  llvm::SmallVector<size_t, 8> worklist{members.front()};
  std::set<size_t> reached{members.front()};
  while (!worklist.empty()) {
    size_t current = worklist.pop_back_val();
    for (const auto &edge : dag.getEdges()) {
      size_t candidate = dag.getNodes().size();
      if (edge.producer == current)
        candidate = edge.consumer;
      else if (edge.consumer == current)
        candidate = edge.producer;
      if (!llvm::is_contained(members, candidate) ||
          !reached.insert(candidate).second)
        continue;
      worklist.push_back(candidate);
    }
  }
  return reached.size() == members.size();
}

void enumerateReferenceLabels(const StructuredDAGAnalysis &dag, size_t index,
                              llvm::SmallVectorImpl<uint32_t> &labels,
                              std::set<CoupledRegionAssignment> &result) {
  if (index == dag.getNodes().size()) {
    uint32_t groupCount = *llvm::max_element(labels) + 1;
    for (uint32_t group = 0; group < groupCount; ++group)
      if (!isConnectedGroup(dag, labels, group))
        return;
    CoupledRegionAssignment assignment;
    for (uint32_t group = 0; group < groupCount; ++group) {
      CoupledRegionGroup entry;
      entry.tile = TileId(0);
      for (auto [node, label] : llvm::enumerate(labels))
        if (label == group)
          entry.nodes.push_back(static_cast<uint32_t>(node));
      assignment.groups.push_back(std::move(entry));
    }
    result.insert(std::move(assignment));
    return;
  }
  uint32_t maximum = 0;
  for (uint32_t label : labels)
    maximum = std::max(maximum, label);
  for (uint32_t label = 0; label <= maximum + (labels.empty() ? 0 : 1);
       ++label) {
    labels.push_back(label);
    enumerateReferenceLabels(dag, index + 1, labels, result);
    labels.pop_back();
  }
}

std::set<CoupledRegionAssignment>
getReference(const StructuredDAGAnalysis &dag) {
  std::set<CoupledRegionAssignment> result;
  llvm::SmallVector<uint32_t, 8> labels;
  enumerateReferenceLabels(dag, 0, labels, result);
  return result;
}

std::set<CoupledRegionAssignment>
enumerateDomain(const CoupledRegionDomain &domain) {
  std::set<CoupledRegionAssignment> result;
  std::optional<CoupledRegionAssignment> current = domain.getFirstAssignment();
  while (current) {
    EXPECT_TRUE(domain.contains(*current));
    result.insert(*current);
    auto next = domain.getNextAssignment(*current);
    EXPECT_TRUE(mlir::succeeded(next));
    if (mlir::failed(next))
      return {};
    current = *next;
  }
  return result;
}

llvm::SmallVector<StructuredDAGNodePlacement, 8>
makeSingleTilePlacements(const StructuredDAGAnalysis &dag) {
  llvm::SmallVector<StructuredDAGNodePlacement, 8> placements;
  for (const auto &node : dag.getNodes()) {
    auto tiling = mlir::cast<mlir::TilingInterface>(node.operation);
    placements.push_back(StructuredDAGNodePlacement{
        node.id,
        llvm::SmallVector<uint32_t, 4>(tiling.getLoopIteratorTypes().size(), 1),
        {TileId(0)}});
  }
  return placements;
}

TEST_F(CoupledRegionTest, DomainMatchesIndependentConnectedPartitionReference) {
  constexpr llvm::StringLiteral chain = R"mlir(
module {
  func.func @chain(%input: tensor<2xf16>) -> tensor<2xf16> {
    %e0 = tensor.empty() : tensor<2xf16>
    %a = linalg.map ins(%input : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2xf16>
    %b = linalg.map ins(%a : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2xf16>
    %c = linalg.map ins(%b : tensor<2xf16>) outs(%e2 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    return %c : tensor<2xf16>
  }
})mlir";
  constexpr llvm::StringLiteral fanin = R"mlir(
module {
  func.func @fanin(%lhs: tensor<2xf16>, %rhs: tensor<2xf16>) -> tensor<2xf16> {
    %e0 = tensor.empty() : tensor<2xf16>
    %a = linalg.map ins(%lhs : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2xf16>
    %b = linalg.map ins(%rhs : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2xf16>
    %c = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<2xf16>, tensor<2xf16>)
        outs(%e2 : tensor<2xf16>) {
      ^bb0(%x: f16, %y: f16, %old: f16):
        %sum = arith.addf %x, %y : f16
        linalg.yield %sum : f16
    } -> tensor<2xf16>
    return %c : tensor<2xf16>
  }
})mlir";
  constexpr llvm::StringLiteral fanout = R"mlir(
module {
  func.func @fanout(%input: tensor<2xf16>)
      -> (tensor<2xf16>, tensor<2xf16>) {
    %e0 = tensor.empty() : tensor<2xf16>
    %a = linalg.map ins(%input : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2xf16>
    %b = linalg.map ins(%a : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2xf16>
    %c = linalg.map ins(%a : tensor<2xf16>) outs(%e2 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    return %b, %c : tensor<2xf16>, tensor<2xf16>
  }
})mlir";
  constexpr llvm::StringLiteral diamond = R"mlir(
module {
  func.func @diamond(%input: tensor<2xf16>) -> tensor<2xf16> {
    %e0 = tensor.empty() : tensor<2xf16>
    %a = linalg.map ins(%input : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2xf16>
    %b = linalg.map ins(%a : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2xf16>
    %c = linalg.map ins(%a : tensor<2xf16>) outs(%e2 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e3 = tensor.empty() : tensor<2xf16>
    %d = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%b, %c : tensor<2xf16>, tensor<2xf16>)
        outs(%e3 : tensor<2xf16>) {
      ^bb0(%x: f16, %y: f16, %old: f16):
        %sum = arith.addf %x, %y : f16
        linalg.yield %sum : f16
    } -> tensor<2xf16>
    return %d : tensor<2xf16>
  }
})mlir";
  constexpr llvm::StringLiteral disconnected = R"mlir(
module {
  func.func @disconnected(%lhs: tensor<2xf16>, %rhs: tensor<2xf16>)
      -> (tensor<2xf16>, tensor<2xf16>) {
    %e0 = tensor.empty() : tensor<2xf16>
    %a = linalg.map ins(%lhs : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2xf16>
    %b = linalg.map ins(%rhs : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    return %a, %b : tensor<2xf16>, tensor<2xf16>
  }
})mlir";

  for (llvm::StringRef source : {chain, fanin, fanout, diamond, disconnected}) {
    auto module = parse(source);
    ASSERT_TRUE(module);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function, &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto placements = makeSingleTilePlacements(*dag);
    auto closed =
        wafer::test::buildTestSpatialDemand(*dag, placements, &failureReason);
    ASSERT_TRUE(mlir::succeeded(closed)) << failureReason;
    auto domain = CoupledRegionDomain::create(
        *dag, closed->spatial, closed->demand, &failureReason);
    ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
    EXPECT_EQ(enumerateDomain(*domain), getReference(*dag));
  }
}

TEST_F(CoupledRegionTest, NonlocalExactDemandRemainsACut) {
  auto module = parse(R"mlir(
module {
  func.func @chain(%input: tensor<4xf16>) -> tensor<4xf16> {
    %e0 = tensor.empty() : tensor<4xf16>
    %a = linalg.map ins(%input : tensor<4xf16>) outs(%e0 : tensor<4xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<4xf16>
    %b = linalg.map ins(%a : tensor<4xf16>) outs(%e1 : tensor<4xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    return %b : tensor<4xf16>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements{
      {0, {2}, {TileId(0), TileId(1)}},
      {1, {2}, {TileId(1), TileId(0)}}};
  auto closed =
      wafer::test::buildTestSpatialDemand(*dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(closed)) << failureReason;
  auto domain = CoupledRegionDomain::create(
      *dag, closed->spatial, closed->demand, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  std::set<CoupledRegionAssignment> assignments = enumerateDomain(*domain);
  ASSERT_EQ(assignments.size(), 1u);
  EXPECT_EQ(assignments.begin()->groups.size(), 4u);
  for (const CoupledRegionGroup &group : assignments.begin()->groups)
    EXPECT_EQ(group.nodes.size(), 1u);
}

TEST_F(CoupledRegionTest, PartialReductionBoundaryRemainsACut) {
  auto module = parse(R"mlir(
module {
  func.func @partial(%input: tensor<2x4xf16>, %init: tensor<2xf16>)
      -> tensor<2xf16> {
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0)>],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<2x4xf16>) outs(%init : tensor<2xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<2xf16>
    %empty = tensor.empty() : tensor<2xf16>
    %mapped = linalg.map ins(%sum : tensor<2xf16>)
        outs(%empty : tensor<2xf16>) (%value: f16) {
      linalg.yield %value : f16
    }
    return %mapped : tensor<2xf16>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 2> placements{
      {0, {1, 2}, {TileId(0), TileId(1)}},
      {1, {2}, {TileId(0), TileId(1)}}};
  auto closed =
      wafer::test::buildTestSpatialDemand(*dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(closed)) << failureReason;
  auto domain = CoupledRegionDomain::create(
      *dag, closed->spatial, closed->demand, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  std::set<CoupledRegionAssignment> assignments = enumerateDomain(*domain);
  ASSERT_EQ(assignments.size(), 1u);
  EXPECT_EQ(assignments.begin()->groups.size(), 4u);
}

TEST_F(CoupledRegionTest, CardDomainIsCartesianAcrossIndependentTiles) {
  auto module = parse(R"mlir(
module {
  func.func @branches(%lhs: tensor<2xf16>, %rhs: tensor<2xf16>)
      -> (tensor<2xf16>, tensor<2xf16>) {
    %e0 = tensor.empty() : tensor<2xf16>
    %a0 = linalg.map ins(%lhs : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2xf16>
    %a1 = linalg.map ins(%a0 : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2xf16>
    %b0 = linalg.map ins(%rhs : tensor<2xf16>) outs(%e2 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e3 = tensor.empty() : tensor<2xf16>
    %b1 = linalg.map ins(%b0 : tensor<2xf16>) outs(%e3 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    return %a1, %b1 : tensor<2xf16>, tensor<2xf16>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  llvm::SmallVector<StructuredDAGNodePlacement, 4> placements{
      {0, {1}, {TileId(0)}},
      {1, {1}, {TileId(0)}},
      {2, {1}, {TileId(1)}},
      {3, {1}, {TileId(1)}}};
  auto closed =
      wafer::test::buildTestSpatialDemand(*dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(closed)) << failureReason;
  auto domain = CoupledRegionDomain::create(
      *dag, closed->spatial, closed->demand, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  std::set<CoupledRegionAssignment> assignments = enumerateDomain(*domain);
  EXPECT_EQ(assignments.size(), 4u);
  std::set<size_t> groupCounts;
  for (const CoupledRegionAssignment &assignment : assignments)
    groupCounts.insert(assignment.groups.size());
  EXPECT_EQ(groupCounts, (std::set<size_t>{2, 3, 4}));
}

TEST_F(CoupledRegionTest, DisconnectedComponentsDoNotScanBellPartitions) {
  constexpr unsigned nodeCount = 8;
  std::string source;
  llvm::raw_string_ostream os(source);
  os << "module {\n  func.func @independent(";
  for (unsigned index = 0; index < nodeCount; ++index) {
    if (index)
      os << ", ";
    os << "%arg" << index << ": tensor<1xf16>";
  }
  os << ") -> (";
  for (unsigned index = 0; index < nodeCount; ++index) {
    if (index)
      os << ", ";
    os << "tensor<1xf16>";
  }
  os << ") {\n";
  for (unsigned index = 0; index < nodeCount; ++index) {
    os << "    %empty" << index << " = tensor.empty() : tensor<1xf16>\n"
       << "    %result" << index << " = linalg.map ins(%arg" << index
       << " : tensor<1xf16>) outs(%empty" << index
       << " : tensor<1xf16>) (%v: f16) { linalg.yield %v : f16 }\n";
  }
  os << "    return ";
  for (unsigned index = 0; index < nodeCount; ++index) {
    if (index)
      os << ", ";
    os << "%result" << index;
  }
  os << " : ";
  for (unsigned index = 0; index < nodeCount; ++index) {
    if (index)
      os << ", ";
    os << "tensor<1xf16>";
  }
  os << "\n  }\n}\n";
  os.flush();

  auto module = parse(source);
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto placements = makeSingleTilePlacements(*dag);
  auto closed =
      wafer::test::buildTestSpatialDemand(*dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(closed)) << failureReason;
  auto domain = CoupledRegionDomain::create(
      *dag, closed->spatial, closed->demand, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;
  CoupledRegionAssignment first = domain->getFirstAssignment();
  EXPECT_EQ(first.groups.size(), nodeCount);
  auto end = domain->getNextAssignment(first);
  ASSERT_TRUE(mlir::succeeded(end));
  EXPECT_FALSE(*end);
}

TEST_F(CoupledRegionTest,
       InterleavedDisconnectedComponentsKeepCanonicalGroupOrder) {
  auto module = parse(R"mlir(
module {
  func.func @interleaved(%lhs: tensor<2xf16>, %rhs: tensor<2xf16>)
      -> (tensor<2xf16>, tensor<2xf16>) {
    %e0 = tensor.empty() : tensor<2xf16>
    %a0 = linalg.map ins(%lhs : tensor<2xf16>) outs(%e0 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e1 = tensor.empty() : tensor<2xf16>
    %b0 = linalg.map ins(%rhs : tensor<2xf16>) outs(%e1 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e2 = tensor.empty() : tensor<2xf16>
    %a1 = linalg.map ins(%a0 : tensor<2xf16>) outs(%e2 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    %e3 = tensor.empty() : tensor<2xf16>
    %b1 = linalg.map ins(%b0 : tensor<2xf16>) outs(%e3 : tensor<2xf16>)
        (%v: f16) { linalg.yield %v : f16 }
    return %a1, %b1 : tensor<2xf16>, tensor<2xf16>
  }
})mlir");
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto placements = makeSingleTilePlacements(*dag);
  auto closed =
      wafer::test::buildTestSpatialDemand(*dag, placements, &failureReason);
  ASSERT_TRUE(mlir::succeeded(closed)) << failureReason;
  auto domain = CoupledRegionDomain::create(
      *dag, closed->spatial, closed->demand, &failureReason);
  ASSERT_TRUE(mlir::succeeded(domain)) << failureReason;

  CoupledRegionAssignment first = domain->getFirstAssignment();
  EXPECT_TRUE(llvm::is_sorted(first.groups));
  EXPECT_TRUE(domain->contains(first));
  EXPECT_EQ(enumerateDomain(*domain).size(), 4u);
}

} // namespace
