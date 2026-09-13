//===- TemporalProposalsTest.cpp --------------------------------------===//

#include "Wafer/Driver/PhysicalDataflow/TemporalProposals.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace {
using namespace wafer;
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  wafer::registerWaferCoreDialects(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::tensor::registerTilingInterfaceExternalModels(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context,
                                        llvm::StringRef body,
                                        llvm::StringRef argumentType,
                                        llvm::StringRef resultType) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module {\n"
            "  wafer.tile.module card_id = 0 tile_id = 0 {\n"
            "    func.func @entry(%input: "
         << argumentType << ") -> " << resultType
         << " {\n"
            "      %result = wafer.tile.region(%input : "
         << argumentType << ") -> (" << resultType
         << ") {\n"
            "      ^bb0(%arg: "
         << argumentType << "):\n"
         << body << "\n        wafer.tile.yield %value : " << resultType
         << "\n      }\n"
            "      return %result : "
         << resultType
         << "\n    }\n"
            "  }\n"
            "}\n";
  return mlir::parseSourceString<mlir::ModuleOp>(stream.str(),
                                                 mlir::ParserConfig(&context));
}

mlir::OwningOpRef<mlir::ModuleOp> makePointwise(mlir::MLIRContext &context,
                                                int64_t rows, int64_t columns) {
  std::string type = "tensor<1x" + std::to_string(rows) + "x" +
                     std::to_string(columns) + "xf16>";
  std::string body;
  llvm::raw_string_ostream out(body);
  out << "%empty = tensor.empty() : " << type << "\n"
      << "%value = linalg.generic {indexing_maps = ["
         "affine_map<(b, m, n) -> (b, m, n)>,"
         "affine_map<(b, m, n) -> (b, m, n)>],"
         "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]}"
      << " ins(%arg : " << type << ") outs(%empty : " << type << ") {"
      << "^bb0(%element: f16, %old: f16): linalg.yield %element : f16"
      << "} -> " << type;
  return parse(context, body, type, type);
}

TileRegionOp regionOf(mlir::ModuleOp module) {
  TileRegionOp result;
  module.walk([&](TileRegionOp region) { result = region; });
  return result;
}

TEST(TemporalProposalsTest, CapacityHalvesAxesAndRetainsOriginalSiblings) {
  auto context = createContext();
  for (int64_t extent : {1024, 1025, 1031}) {
    auto left = makePointwise(*context, extent, 64);
    auto right = makePointwise(*context, extent, 64);
    ASSERT_TRUE(left && right);
    auto a = buildTemporalDomain(regionOf(*left));
    auto b = buildTemporalDomain(regionOf(*right));
    ASSERT_TRUE(a.succeeded() && b.succeeded());
    TemporalProposals proposals({&*a.domain, &*b.domain});
    std::vector<TemporalChoice> initial{
        *a.domain->getFirstChoice().getChoice(),
        *b.domain->getFirstChoice().getChoice()};
    ASSERT_TRUE(proposals.visitRaw(initial));
    ASSERT_TRUE(proposals.observeCapacity(initial, {{0, 0, 1}, {0, 0, 2}}));
    std::set<std::pair<int64_t, int64_t>> observed;
    while (proposals.prepareNext(TemporalProposalKind::Repair)) {
      auto point = proposals.take(TemporalProposalKind::Repair);
      const auto &sizes = point[0].scopes[0].iteratorTileSizes;
      EXPECT_TRUE(observed.emplace(sizes[1], sizes[2]).second);
      EXPECT_EQ(point[1], initial[1]);
      EXPECT_TRUE(a.domain->contains(point[0]));
      EXPECT_FALSE(proposals.visitRaw(point));
    }
    const std::set<std::pair<int64_t, int64_t>> expected{
        {extent / 2, 32}, {extent / 2, 64}, {extent, 32}};
    EXPECT_EQ(observed, expected);
    // Missing provenance is not permission to shrink another owner.
    EXPECT_FALSE(proposals.observeCapacity(initial, {}));
    EXPECT_FALSE(proposals.prepareNext(TemporalProposalKind::Repair));
  }
}

TEST(TemporalProposalsTest,
     PreparingTheLastCoarseSiblingDoesNotReleaseItsSlot) {
  auto context = createContext();
  auto module = makePointwise(*context, 1025, 1031);
  auto built = buildTemporalDomain(regionOf(*module));
  ASSERT_TRUE(built.succeeded());
  TemporalProposals proposals({&*built.domain});
  std::vector<TemporalChoice> initial{
      *built.domain->getFirstChoice().getChoice()};
  ASSERT_TRUE(proposals.visitRaw(initial));
  ASSERT_TRUE(proposals.observeCapacity(initial, {{0, 0, 1}, {0, 0, 2}}));
  for (size_t direction = 0; direction < 3; ++direction) {
    ASSERT_TRUE(proposals.prepareNext(TemporalProposalKind::Repair));
    EXPECT_TRUE(proposals.hasCapacityRoundInProgress());
    (void)proposals.take(TemporalProposalKind::Repair);
  }
  EXPECT_FALSE(proposals.hasCapacityRoundInProgress());
}

TEST(TemporalProposalsTest, CapacityNewEvidenceCombinesWithoutLosingAxes) {
  auto context = createContext();
  for (int64_t extent : {1024, 1025, 1031}) {
    auto module = makePointwise(*context, extent, 1031);
    auto built = buildTemporalDomain(regionOf(*module));
    ASSERT_TRUE(built.succeeded());
    TemporalProposals proposals({&*built.domain});
    std::vector<TemporalChoice> initial{
        *built.domain->getFirstChoice().getChoice()};
    ASSERT_TRUE(proposals.visitRaw(initial));
    ASSERT_TRUE(proposals.observeCapacity(initial, {{0, 0, 1}}));
    auto first = proposals.take(TemporalProposalKind::Repair);
    EXPECT_EQ(first[0].scopes[0].iteratorTileSizes[1], extent / 2);
    EXPECT_EQ(first[0].scopes[0].iteratorTileSizes[2], 1031);
    ASSERT_TRUE(proposals.observeCapacity(initial, {{0, 0, 2}}));
    std::set<std::pair<int64_t, int64_t>> observed;
    while (proposals.prepareNext(TemporalProposalKind::Repair)) {
      auto point = proposals.take(TemporalProposalKind::Repair);
      const auto &sizes = point[0].scopes[0].iteratorTileSizes;
      EXPECT_TRUE(observed.emplace(sizes[1], sizes[2]).second);
      EXPECT_TRUE(built.domain->contains(point[0]));
    }
    const std::set<std::pair<int64_t, int64_t>> expected{{extent / 2, 515},
                                                         {extent, 515}};
    EXPECT_EQ(observed, expected);
  }
}

TEST(TemporalProposalsTest, CapacityBatchesScopesAndRetainsMixedChoices) {
  auto context = createContext();
  for (int64_t extent : {1024, 1025, 1031}) {
    auto left = makePointwise(*context, extent, 1031);
    auto right = makePointwise(*context, 1025, extent);
    auto unrelated = makePointwise(*context, 4096, 4096);
    auto a = buildTemporalDomain(regionOf(*left));
    auto b = buildTemporalDomain(regionOf(*right));
    auto c = buildTemporalDomain(regionOf(*unrelated));
    ASSERT_TRUE(a.succeeded() && b.succeeded() && c.succeeded());
    TemporalProposals proposals({&*a.domain, &*b.domain, &*c.domain});
    std::vector<TemporalChoice> initial{
        *a.domain->getFirstChoice().getChoice(),
        *b.domain->getFirstChoice().getChoice(),
        *c.domain->getFirstChoice().getChoice()};
    ASSERT_TRUE(proposals.visitRaw(initial));
    ASSERT_TRUE(proposals.observeCapacity(
        initial, {{0, 0, 1}, {0, 0, 2}, {1, 0, 1}, {1, 0, 2}}));
    auto first = proposals.take(TemporalProposalKind::Repair);
    EXPECT_EQ(first[0].scopes[0].iteratorTileSizes[1], extent / 2);
    EXPECT_EQ(first[0].scopes[0].iteratorTileSizes[2], 515);
    EXPECT_EQ(first[1].scopes[0].iteratorTileSizes[1], 512);
    EXPECT_EQ(first[1].scopes[0].iteratorTileSizes[2], extent / 2);
    EXPECT_EQ(first[2], initial[2]);
    bool mixed = false, independent = false;
    while (proposals.prepareNext(TemporalProposalKind::Repair)) {
      auto point = proposals.take(TemporalProposalKind::Repair);
      EXPECT_EQ(point[2], initial[2]);
      EXPECT_TRUE(a.domain->contains(point[0]));
      EXPECT_TRUE(b.domain->contains(point[1]));
      const auto &x = point[0].scopes[0].iteratorTileSizes;
      const auto &y = point[1].scopes[0].iteratorTileSizes;
      mixed |= x[1] == extent / 2 && x[2] == 1031 && y[1] == 1025 &&
               y[2] == extent / 2;
      independent |=
          x[1] == extent / 2 && x[2] == 1031 && point[1] == initial[1];
    }
    EXPECT_TRUE(mixed && independent);
  }
}

TEST(TemporalProposalsTest, AcceptedNeighborsExpandAndRevisitAnUnprovedGap) {
  auto context = createContext();
  for (int64_t extent : {1024, 1025, 1031}) {
    auto module = makePointwise(*context, extent, 64);
    ASSERT_TRUE(module);
    auto domain = buildTemporalDomain(regionOf(*module));
    ASSERT_TRUE(domain.succeeded());
    TemporalProposals proposals({&*domain.domain});
    auto choice = *domain.domain->getFirstChoice().getChoice();
    choice.scopes[0].iteratorTileSizes[1] = 512;
    choice.scopes[0].loopOrder = *buildFirstTemporalLoopOrder(
        domain.domain->getScopeDescriptors()[0].iterationExtents,
        choice.scopes[0].iteratorTileSizes);
    std::vector<TemporalChoice> initial{choice};
    ASSERT_TRUE(proposals.visitRaw(initial));
    proposals.observeAccepted(initial, 100);
    bool lowerObserved = false, upperObserved = false;
    bool expanded = false, gap = false;
    while (proposals.prepareNext(TemporalProposalKind::Improve)) {
      auto point = proposals.take(TemporalProposalKind::Improve);
      auto sizes = point[0].scopes[0].iteratorTileSizes;
      if (sizes[1] == 511 && sizes[2] == 64 && !lowerObserved) {
        lowerObserved = true;
        proposals.observeAccepted(point, 99);
      }
      upperObserved |= sizes[1] == 513 && sizes[2] == 64;
      if (sizes[1] == 509 && sizes[2] == 64) {
        expanded = true;
        proposals.observeAccepted(point, 101);
      }
      gap |= sizes[1] == 510 && sizes[2] == 64;
    }
    EXPECT_TRUE(lowerObserved && upperObserved);
    EXPECT_TRUE(expanded);
    EXPECT_TRUE(gap);
  }
}

TEST(TemporalProposalsTest,
     AcceptedPollBatchesScopesAndExchangesIndependentAxes) {
  auto context = createContext();
  for (int64_t extent : {1024, 1025, 1031}) {
    auto left = makePointwise(*context, extent, 1031);
    auto right = makePointwise(*context, 1025, extent);
    auto a = buildTemporalDomain(regionOf(*left));
    auto b = buildTemporalDomain(regionOf(*right));
    ASSERT_TRUE(a.succeeded() && b.succeeded());
    std::vector<TemporalChoice> initial{
        *a.domain->getFirstChoice().getChoice(),
        *b.domain->getFirstChoice().getChoice()};
    for (auto &choice : initial)
      for (auto &scope : choice.scopes) {
        scope.iteratorTileSizes[1] = 256;
        scope.iteratorTileSizes[2] = 512;
        scope.loopOrder = {1, 2};
      }
    ASSERT_TRUE(a.domain->contains(initial[0]));
    ASSERT_TRUE(b.domain->contains(initial[1]));
    TemporalProposals proposals({&*a.domain, &*b.domain});
    ASSERT_TRUE(proposals.visitRaw(initial));
    proposals.observeAccepted(initial, 100);
    auto lower = proposals.take(TemporalProposalKind::Improve);
    auto upper = proposals.take(TemporalProposalKind::Improve);
    for (size_t domain = 0; domain < 2; ++domain) {
      EXPECT_EQ(lower[domain].scopes[0].iteratorTileSizes[1], 128);
      EXPECT_EQ(lower[domain].scopes[0].iteratorTileSizes[2], 256);
      EXPECT_EQ(upper[domain].scopes[0].iteratorTileSizes[1], 384);
      EXPECT_EQ(upper[domain].scopes[0].iteratorTileSizes[2], 768);
    }
    bool exchange = false, order = false, independent = false;
    size_t visited = 2;
    while (proposals.prepareNext(TemporalProposalKind::Improve)) {
      auto point = proposals.take(TemporalProposalKind::Improve);
      ASSERT_LT(++visited, 1000u); // Finite neighborhood; no feedback is added.
      EXPECT_TRUE(a.domain->contains(point[0]));
      EXPECT_TRUE(b.domain->contains(point[1]));
      const auto &sizes = point[0].scopes[0].iteratorTileSizes;
      exchange |= sizes[1] == 128 && sizes[2] == 768 && point[1] == initial[1];
      independent |=
          sizes[1] == 128 && sizes[2] == 512 && point[1] == initial[1];
      order |= sizes == initial[0].scopes[0].iteratorTileSizes &&
               point[0].scopes[0].loopOrder != initial[0].scopes[0].loopOrder;
    }
    EXPECT_TRUE(exchange && order && independent);
  }
}

TEST(TemporalProposalsTest, CapacityCannotShrinkAnUnrelatedLargerDimension) {
  auto context = createContext();
  for (int64_t extent : {1024, 1025, 1031}) {
    auto module = makePointwise(*context, extent, 64);
    ASSERT_TRUE(module);
    auto domain = buildTemporalDomain(regionOf(*module));
    ASSERT_TRUE(domain.succeeded());
    TemporalProposals proposals({&*domain.domain});
    std::vector<TemporalChoice> initial{
        *domain.domain->getFirstChoice().getChoice()};
    ASSERT_TRUE(proposals.visitRaw(initial));
    ASSERT_TRUE(proposals.observeCapacity(initial, {{0, 0, 2}}));
    auto repaired = proposals.take(TemporalProposalKind::Repair);
    EXPECT_EQ(repaired[0].scopes[0].iteratorTileSizes[1], extent);
    EXPECT_EQ(repaired[0].scopes[0].iteratorTileSizes[2], 32);
    EXPECT_TRUE(domain.domain->contains(repaired[0]));
  }
}

TEST(TemporalProposalsTest,
     StructuralStrataChangeOrderWithoutDroppingFamilies) {
  auto context = createContext();
  for (int64_t extent : {1024, 1025, 1031}) {
    auto module = makePointwise(*context, extent, 64);
    ASSERT_TRUE(module);
    auto domain = buildTemporalDomain(regionOf(*module));
    ASSERT_TRUE(domain.succeeded());
    auto initial = *domain.domain->getFirstChoice().getChoice();
    auto independent = *domain.domain->getFirstIndependentChoice().getChoice();
    std::vector<TemporalChoice> reference;
    for (size_t stratum : {0, 1, 4, 100}) {
      TemporalProposals proposals({&*domain.domain});
      proposals.seed({initial}, stratum);
      proposals.seed({independent}, stratum);
      std::vector<TemporalChoice> points;
      while (proposals.prepareNext(TemporalProposalKind::Explore)) {
        auto point = proposals.take(TemporalProposalKind::Explore)[0];
        EXPECT_TRUE(domain.domain->contains(point));
        EXPECT_FALSE(llvm::is_contained(points, point));
        points.push_back(point);
      }
      ASSERT_GE(points.size(), 2u);
      EXPECT_EQ(points[0].kind, TemporalTraversalKind::Joint);
      EXPECT_EQ(points[1].kind, TemporalTraversalKind::Independent);
      EXPECT_EQ(points[0], initial);
      EXPECT_FALSE(points[1] == independent);
      EXPECT_TRUE(llvm::is_contained(points, independent));
      if (!stratum) {
        EXPECT_EQ(points[0], initial);
        reference = points;
      } else {
        ASSERT_EQ(reference.size(), points.size());
        for (const auto &point : reference)
          EXPECT_TRUE(llvm::is_contained(points, point));
        if (stratum == 1 || stratum == 4) {
          EXPECT_FALSE(reference[2] == points[2]);
        }
      }
    }
  }
}

TEST(TemporalProposalsTest,
     FullOnlyKernelExtentsSupplyDistantChoicesWithoutRestrictingNeighbors) {
  auto context = createContext();
  auto module = parse(*context,
                      R"mlir(
        %key = tensor.empty() : tensor<2x4x1031x64xf16>
        %value_input = tensor.empty() : tensor<2x4x1031x128xf16>
        %scale = arith.constant 1.0 : f32
        %accumulator = tensor.empty() : tensor<2x4x1025x128xf16>
        %maximum = tensor.empty() : tensor<2x4x1025xf32>
        %sum = tensor.empty() : tensor<2x4x1025xf32>
        %value, %next_maximum, %next_sum =
            wafer.linalg_ext.online_attention
            ins(%arg, %key, %value_input, %scale : tensor<2x4x1025x64xf16>,
                tensor<2x4x1031x64xf16>, tensor<2x4x1031x128xf16>, f32)
            outs(%accumulator, %maximum, %sum : tensor<2x4x1025x128xf16>,
                tensor<2x4x1025xf32>, tensor<2x4x1025xf32>)
            indexing_maps = [
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>,
              affine_map<(b, h, m, k1, k2, n) -> ()>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>] score {
            ^bb0(%attention_0_dot: f16, %attention_0_scale: f32):
              %attention_0_converted = arith.extf %attention_0_dot : f16 to f32
              %attention_0_scaled = arith.mulf %attention_0_converted, %attention_0_scale : f32
              wafer.linalg_ext.attention.yield %attention_0_scaled : f32
            }
            -> (tensor<2x4x1025x128xf16>, tensor<2x4x1025xf32>,
                tensor<2x4x1025xf32>))mlir",
                      "tensor<2x4x1025x64xf16>", "tensor<2x4x1025x128xf16>");

  ASSERT_TRUE(module);
  auto built = buildTemporalDomain(regionOf(*module));
  ASSERT_TRUE(built.succeeded());
  auto initial = *built.domain->getFirstChoice().getChoice();
  TemporalProposals proposals({&*built.domain});
  proposals.seed({initial});
  EXPECT_EQ(proposals.take(TemporalProposalKind::Explore)[0], initial);
  auto point = proposals.take(TemporalProposalKind::Explore);
  const auto &sizes = point[0].scopes[0].iteratorTileSizes;
  EXPECT_EQ(sizes[2], 128);
  EXPECT_EQ(sizes[4], 128);
  EXPECT_EQ(sizes[3], 64);
  EXPECT_EQ(sizes[5], 128);
  proposals.observeAccepted(point, 100);
  bool has127 = false, has129 = false;
  while (proposals.prepareNext(TemporalProposalKind::Improve)) {
    auto neighbor = proposals.take(TemporalProposalKind::Improve)[0];
    EXPECT_TRUE(built.domain->contains(neighbor));
    auto next = neighbor.scopes[0].iteratorTileSizes;
    EXPECT_EQ(next[3], 64);
    EXPECT_EQ(next[5], 128);
    has127 |= next[2] == 127;
    has129 |= next[2] == 129;
  }
  EXPECT_TRUE(has127 && has129);
}

TEST(TemporalProposalsTest,
     CoupledSeedsReachSharedTraversalWithWholeBroadcastAxis) {
  for (int64_t extent : {1024, 1025, 1031})
    for (bool permuted : {false, true})
      for (bool extraConsumer : {false, true}) {
        auto context = createContext();
        std::string m = std::to_string(extent);
        std::string input = "tensor<1x2x" + m + "x1031xf16>";
        std::string state = "tensor<1x2x" + m + "xf16>";
        std::string output =
            "tensor<1x2x" + (permuted ? "128x" + m : m + "x128") + "xf16>";
        std::string text;
        llvm::raw_string_ostream body(text);
        body << "%z = arith.constant 0.0 : f16\n%e = tensor.empty() : " << state
             << "\n%i = linalg.fill ins(%z : f16) outs(%e : " << state
             << ") -> " << state
             << "\n%p:3 = linalg.generic {indexing_maps = "
                "[affine_map<(b,h,m,k)->(b,h,m,k)>,affine_map<(b,h,m,k)->(b,h,"
                "m)>,"
                "affine_map<(b,h,m,k)->(b,h,m)>,"
                "affine_map<(b,h,m,k)->(b,h,m)>], iterator_types = "
                "[\"parallel\",\"parallel\",\"parallel\",\"reduction\"]} "
                "ins(%arg : "
             << input << ") outs(%i, %i, %i : " << state << ", " << state
             << ", " << state
             << ") { ^bb0(%x: f16, %a: f16, %b: f16, %unused: f16): "
                "%s = arith.addf %x, %a : f16 "
                "%t = arith.maximumf %x, %b : f16 "
                "linalg.yield %s, %t, %s : f16, f16, f16 } -> ("
             << state << ", " << state << ", " << state << ")\n";
        if (extraConsumer)
          body << "%other = linalg.add ins(%p#0, %p#1 : " << state << ", "
               << state << ") outs(%e : " << state << ") -> " << state << "\n";
        const char *order = permuted ? "b,h,c,m" : "b,h,m,c";
        body << "%out = tensor.empty() : " << output
             << "\n%value = linalg.generic {indexing_maps = [affine_map<("
             << order << ")->(b,h,m)>,affine_map<(" << order
             << ")->(b,h,m)>,affine_map<(" << order << ")->(" << order
             << ")>], iterator_types = "
                "[\"parallel\",\"parallel\",\"parallel\",\"parallel\"]}"
                " ins(%p#0, %p#1 : "
             << state << ", " << state << ") outs(%out : " << output
             << ") { ^bb0(%a: f16, %b: f16, %old: f16): "
                "%sum = arith.addf %a, %b : f16 "
                "linalg.yield %sum : f16 } -> "
             << output;
        auto module = parse(*context, body.str(), input, output);
        ASSERT_TRUE(module);
        auto region = regionOf(*module);
        auto built = buildTemporalDomain(region);
        ASSERT_TRUE(built.succeeded());

        const auto initial =
            *built.domain->getFirstIndependentChoice().getChoice();
        TemporalProposals proposals({&*built.domain});
        proposals.seed({initial});
        bool coordinated = false;
        while (proposals.prepareNext(TemporalProposalKind::Explore)) {
          auto point = proposals.take(TemporalProposalKind::Explore)[0];
          EXPECT_TRUE(built.domain->contains(point));
          if (point.scopes.size() != 2)
            continue;
          auto &producer = point.scopes[0].iteratorTileSizes;
          auto &consumer = point.scopes[1].iteratorTileSizes;
          coordinated |= producer[2] < extent && producer[3] < 1031 &&
                         producer[2] == consumer[permuted ? 3 : 2] &&
                         consumer[permuted ? 2 : 3] == 128;
        }
        EXPECT_EQ(coordinated, !extraConsumer);
      }
}

TEST(TemporalProposalsTest, RawOracleRetainsOrdersTailsAndCombinationOptimum) {
  // Tiny 1x3x4 bounds the exhaustive oracle; real-size neighbor and actual
  // Instr/SPM feedback coverage is separate. Capacity is deliberately
  // nonmonotone here: only a diagonal pair improves over the initial point.
  auto context = createContext();
  auto module = makePointwise(*context, 3, 4);
  ASSERT_TRUE(module);
  auto domain = buildTemporalDomain(regionOf(*module));
  ASSERT_TRUE(domain.succeeded());
  auto run = [&]() {
    TemporalProposals proposals({&*domain.domain});
    auto first = domain.domain->getFirstChoice();
    proposals.seed({*first.getChoice()});
    std::vector<TemporalChoice> observed;
    uint64_t best = 1000;
    auto evaluate = [&](std::vector<TemporalChoice> point) {
      EXPECT_TRUE(domain.domain->contains(point[0]));
      EXPECT_FALSE(llvm::is_contained(observed, point[0]));
      observed.push_back(point[0]);
      auto sizes = point[0].scopes[0].iteratorTileSizes;
      uint64_t cost = sizes[1] == 2 && sizes[2] == 3   ? 1
                      : sizes[1] == 3 && sizes[2] == 4 ? 100
                                                       : 200;
      best = std::min(best, cost);
      if (sizes[1] == 1 && sizes[2] % 2 == 0)
        proposals.observeCapacity(point, {{0, 0, 1}, {0, 0, 2}});
      else
        proposals.observeAccepted(point, cost);
    };
    for (unsigned round = 0; round < 1000; ++round) {
      bool any = false;
      for (auto kind :
           {TemporalProposalKind::Explore, TemporalProposalKind::Repair,
            TemporalProposalKind::Improve})
        if (proposals.prepareNext(kind)) {
          evaluate(proposals.take(kind));
          any = true;
        }
      if (!any)
        break;
    }
    EXPECT_EQ(best, 1u);
    auto raw = domain.domain->getFirstChoice();
    size_t count = 0;
    while (raw.getKind() == TemporalSuccessorKind::Choice) {
      ++count;
      if (proposals.visitRaw({*raw.getChoice()}))
        observed.push_back(*raw.getChoice());
      raw = domain.domain->getNextChoice(*raw.getCursor());
    }
    EXPECT_EQ(raw.getKind(), TemporalSuccessorKind::End);
    EXPECT_EQ(observed.size(), count);
    return observed;
  };
  EXPECT_EQ(run(), run());
}

} // namespace
