//===- OnlineAttentionDecompositionTest.cpp ---------------------------===//

#include "Wafer/Transforms/Linalg/OnlineAttentionDecomposition.h"
#include "Wafer/Transforms/Linalg/AttentionVisibility.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"

#include "Wafer/Analysis/Tile/TileDataflowAnalysis.h"
#include "Wafer/Conversion/TileToInstr/TileToInstr.h"
#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/Instr/NCCJoinPlacement.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <functional>
#include <memory>
#include <string>
#include <tuple>

namespace {

using namespace wafer;
using namespace wafer::compiler::detail;

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  registerCompilationDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

std::string tensorType(llvm::StringRef shape, llvm::StringRef elementType) {
  return "tensor<" + shape.str() + "x" + elementType.str() + ">";
}

mlir::OwningOpRef<mlir::ModuleOp> parseOnlineModule(mlir::MLIRContext &context,
                                                    int64_t keyValueExtent,
                                                    llvm::StringRef elementType,
                                                    bool withMask) {
  const std::string queryType = tensorType("2x4x1025x64", elementType);
  const std::string keyType =
      tensorType("2x4x" + std::to_string(keyValueExtent) + "x64", elementType);
  const std::string valueType =
      tensorType("2x4x" + std::to_string(keyValueExtent) + "x128", elementType);
  const std::string accumulatorType = tensorType("2x4x1025x128", "f32");
  const std::string maskType =
      "tensor<2x1025x" + std::to_string(keyValueExtent) + "xf32>";
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << R"mlir(module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%query: )mlir"
         << queryType << ", %key: " << keyType
         << ", %value_input: " << valueType;
  if (withMask)
    stream << ", %mask: " << maskType;
  stream << ") -> " << accumulatorType << R"mlir( {
      %result = wafer.tile.region(%query, %key, %value_input)mlir";
  if (withMask)
    stream << ", %mask";
  stream << " : " << queryType << ", " << keyType << ", " << valueType;
  if (withMask)
    stream << ", " << maskType;
  stream << ") -> (" << accumulatorType << R"mlir() {
      ^bb0(%query_arg: )mlir"
         << queryType << ", %key_arg: " << keyType
         << ", %value_arg: " << valueType;
  if (withMask)
    stream << ", %mask_arg: " << maskType;
  stream << R"mlir():
        %scale = arith.constant 1.0 : f32
        %empty_accumulator = tensor.empty() : )mlir"
         << accumulatorType << "\n%zero_accumulator = arith.constant 0.0 : f32"
         << "\n%accumulator = linalg.fill ins(%zero_accumulator : "
         << "f32) outs(%empty_accumulator : " << accumulatorType << ") -> "
         << accumulatorType << R"mlir(
        %empty_maximum = tensor.empty() : tensor<2x4x1025xf32>
        %empty_sum = tensor.empty() : tensor<2x4x1025xf32>
        %minus_inf = arith.constant 0xFF800000 : f32
        %zero_sum = arith.constant 0.0 : f32
        %maximum = linalg.fill ins(%minus_inf : f32) outs(%empty_maximum : tensor<2x4x1025xf32>) -> tensor<2x4x1025xf32>
        %sum = linalg.fill ins(%zero_sum : f32) outs(%empty_sum : tensor<2x4x1025xf32>) -> tensor<2x4x1025xf32>
        %next_accumulator, %next_maximum, %next_sum =
            wafer.linalg_ext.online_attention
            ins(%query_arg, %key_arg, %value_arg, %scale)mlir";
  if (withMask)
    stream << ", %mask_arg";
  stream << " : " << queryType << ", " << keyType << ", " << valueType
         << ", f32";
  if (withMask)
    stream << ", " << maskType;
  stream << R"mlir()
            outs(%accumulator, %maximum, %sum : )mlir"
         << accumulatorType
         << R"mlir(, tensor<2x4x1025xf32>, tensor<2x4x1025xf32>)
            indexing_maps = [
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m, k1)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, k1)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, k2, n)>,
              affine_map<(b, h, m, k1, k2, n) -> ()>,
)mlir";
  if (withMask)
    stream << "              affine_map<(b, h, m, k1, k2, n) -> "
              "(b, m, k2)>,\n";
  stream
      << R"mlir(              affine_map<(b, h, m, k1, k2, n) -> (b, h, m, n)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>,
              affine_map<(b, h, m, k1, k2, n) -> (b, h, m)>]
            score { ^bb0(%dot: )mlir"
      << "f32, %scale_arg: f32";
  if (withMask)
    stream << ", %mask_scalar: f32";
  stream << "):\n";
  stream << "%scaled = arith.mulf %dot, %scale_arg : f32\n";
  if (withMask)
    stream << "%masked = arith.addf %scaled, %mask_scalar : f32\n"
              "wafer.linalg_ext.attention.yield %masked : f32\n";
  else
    stream << "wafer.linalg_ext.attention.yield %scaled : f32\n";
  stream << "} -> (" << accumulatorType
         << R"mlir(, tensor<2x4x1025xf32>, tensor<2x4x1025xf32>)
        wafer.tile.yield %next_accumulator : )mlir"
         << accumulatorType << R"mlir(
      }
      return %result : )mlir"
         << accumulatorType << R"mlir(
    }
  }
}
)mlir";
  return mlir::parseSourceString<mlir::ModuleOp>(stream.str(),
                                                 mlir::ParserConfig(&context));
}

TileRegionOp findRegion(mlir::ModuleOp module) {
  TileRegionOp result;
  module.walk([&](TileRegionOp region) { result = region; });
  return result;
}

template <typename Op> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](Op) { ++count; });
  return count;
}

mlir::LogicalResult lowerPhysicalToInstr(mlir::ModuleOp module) {
  llvm::SmallVector<TileRegionOp, 8> regions;
  module.walk([&](TileRegionOp region) { regions.push_back(region); });
  TileRegionToInstrLoweringSession session(*module.getContext());
  for (TileRegionOp region : regions)
    if (mlir::failed(convertTileRegionToInstr(region, session)))
      return mlir::failure();
  if (mlir::failed(convertBufferizationCopiesToInstr(module, session)) ||
      mlir::failed(rebuildRequiredNCCJoins(module)) ||
      analysis::containsTileDataflowOperations(module.getOperation()))
    return mlir::failure();
  return mlir::verify(module);
}

TemporalChoice selectK2Tile(const TemporalDomain &domain, int64_t tileSize,
                            int64_t queryTileSize = 0) {
  TemporalSuccessor first = domain.getFirstChoice();
  EXPECT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
  TemporalChoice choice = *first.getChoice();
  llvm::ArrayRef<TemporalScopeDescriptor> descriptors =
      domain.getScopeDescriptors();
  EXPECT_EQ(choice.scopes.size(), descriptors.size());
  for (auto [scope, descriptor] : llvm::zip_equal(choice.scopes, descriptors)) {
    auto online =
        mlir::dyn_cast<LinalgExtOnlineAttentionOp>(descriptor.operation);
    if (!online)
      continue;
    auto roles = online.getIterationRoles();
    EXPECT_TRUE(mlir::succeeded(roles));
    for (unsigned dimension : roles->keyValueReduction)
      scope.iteratorTileSizes[dimension] =
          std::min(tileSize, descriptor.iterationExtents[dimension]);
    if (queryTileSize)
      for (unsigned dimension : roles->query)
        scope.iteratorTileSizes[dimension] =
            std::min(queryTileSize, descriptor.iterationExtents[dimension]);
    auto order = buildFirstTemporalLoopOrder(descriptor.iterationExtents,
                                             scope.iteratorTileSizes,
                                             descriptor.precedence);
    EXPECT_TRUE(mlir::succeeded(order));
    if (mlir::succeeded(order))
      scope.loopOrder = std::move(*order);
  }
  EXPECT_TRUE(domain.contains(choice));
  return choice;
}

unsigned countContractions(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](mlir::linalg::LinalgOp operation) {
    if (operation.getNumDpsInputs() == 2 &&
        llvm::is_contained(operation.getIteratorTypesArray(),
                           mlir::utils::IteratorType::reduction))
      ++count;
  });
  return count;
}

unsigned countRowReductions(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](mlir::linalg::LinalgOp operation) {
    if (operation.getNumDpsInputs() == 1 &&
        llvm::is_contained(operation.getIteratorTypesArray(),
                           mlir::utils::IteratorType::reduction))
      ++count;
  });
  return count;
}

// Scalar-only proof fixture: the production tensor path is covered below.
TEST(OnlineAttentionDecompositionTest, BoundaryPatternsFollowActualLoopGrids) {
  for (auto [queryTile, keyTile, queryStart, expectedPatterns] :
       {std::tuple{256, 256, 0, 1}, std::tuple{192, 128, 0, 4},
        std::tuple{256, 256, 5, 2}}) {
    auto context = createContext();
    std::string text = R"mlir(module {
      func.func @entry() {
        %c0 = arith.constant 0 : index
        %start = arith.constant QUERY_START : index
        %c4096 = arith.constant 4096 : index
        %qsize = arith.constant QUERY_TILE : index
        %ksize = arith.constant KEY_TILE : index
        %c1 = arith.constant 1 : index
        scf.for %q = %start to %c4096 step %qsize {
          %qend = arith.addi %q, %qsize : index
          %upper = arith.minsi %c4096, %qend : index
          scf.for %k = %c0 to %upper step %ksize {
            %kend = arith.addi %k, %ksize : index
            %first = arith.addi %q, %c1 : index
            %visible = arith.cmpi ult, %k, %qend : index
            %full = arith.cmpi ule, %kend, %first : index
            scf.if %visible {
              scf.if %full {
              } else {
                %delta = arith.subi %q, %k : index
              }
            }
          }
        }
        return
      }
    })mlir";
    for (auto [name, value] :
         {std::pair{"QUERY_START", queryStart},
          std::pair{"QUERY_TILE", queryTile}, std::pair{"KEY_TILE", keyTile}})
      text.replace(text.find(name), std::string(name).size(),
                   std::to_string(value));
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(module);
    mlir::arith::SubIOp difference;
    module->walk([&](mlir::arith::SubIOp op) { difference = op; });
    ASSERT_TRUE(difference);
    auto keyLoop = difference->getParentOfType<mlir::scf::ForOp>();
    auto limit = keyLoop.getUpperBound().getDefiningOp<mlir::arith::MinSIOp>();
    ASSERT_TRUE(limit);
    module->walk([&](mlir::arith::AddIOp add) {
      if (add.getLhs() == keyLoop.getInductionVar()) {
        EXPECT_TRUE(
            proveAttentionPositionOrder(add.getResult(), limit.getLhs()));
      }
    });
    auto range =
        getAttentionPositionRange(difference.getLhs(), difference.getRhs(),
                                  difference, -queryTile, keyTile - 1);
    EXPECT_EQ(range.size(), expectedPatterns)
        << range.first << ":" << range.last << ":" << range.step;
    for (int64_t q = queryStart; q < 4096; q += queryTile)
      for (int64_t k = 0; k < 4096; k += keyTile) {
        if (k >= q + queryTile || k + keyTile <= q + 1)
          continue;
        EXPECT_GE(q - k, range.first);
        EXPECT_LE(q - k, range.last);
        EXPECT_EQ((q - k - range.first) % range.step, 0);
      }
  }
}

TEST(OnlineAttentionDecompositionTest, PositionBiasHasExactIntegerMaskValues) {
  for (llvm::StringRef dtype : {"f16", "bf16"})
    for (int64_t diagonal : {-73, 0, 37}) {
      auto context = createContext();
      auto module = parseOnlineModule(*context, 128, dtype, false);
      ASSERT_TRUE(module);
      LinalgExtOnlineAttentionOp source;
      module->walk([&](LinalgExtOnlineAttentionOp op) { source = op; });
      mlir::OpBuilder builder(source);
      auto loc = source.getLoc();
      // Absolute positions exceed F32's exact integer range. The score is F32,
      // but no coordinate is represented in that type.
      const int64_t base = INT64_C(1) << 40;
      auto query =
          builder.create<mlir::arith::ConstantIndexOp>(loc, base + diagonal);
      auto key = builder.create<mlir::arith::ConstantIndexOp>(loc, base);
      auto end = builder.create<mlir::arith::ConstantIndexOp>(loc, base + 123);
      source.getPositionsMutable().append(mlir::ValueRange{query, key, end});
      source.setCausal(true);
      source.setPositionMapAttr(mlir::AffineMapAttr::get(mlir::AffineMap::get(
          6, 0, {builder.getAffineDimExpr(2), builder.getAffineDimExpr(4)},
          context.get())));
      StructuredMaterializationRelations relations;
      auto region = findRegion(*module);
      relations.structuralOutputs.push_back({0, region.getResult(0)});
      OnlineAttentionDecompositionFailure failure;
      ASSERT_TRUE(mlir::succeeded(
          decomposeOnlineAttention(*module, relations, &failure)))
          << failure.detail;
      unsigned templates = 0;
      module->walk([&](mlir::arith::ConstantOp constant) {
        auto values =
            mlir::dyn_cast<mlir::DenseFPElementsAttr>(constant.getValue());
        if (!values)
          return;
        ASSERT_EQ(values.getType().getShape(),
                  (llvm::ArrayRef<int64_t>{1025, 128}));
        ++templates;
        unsigned mismatches = 0;
        int64_t offset = 0;
        for (llvm::APFloat value : values.getValues<llvm::APFloat>()) {
          int64_t q = offset / 128, k = offset % 128;
          bool masked = k > q + diagonal || k >= 123;
          mismatches += masked ? !(value.isInfinity() && value.isNegative())
                               : !value.isZero();
          ++offset;
        }
        EXPECT_EQ(mismatches, 0u);
      });
      EXPECT_EQ(templates, 1u);
      EXPECT_EQ(countOps<mlir::arith::SIToFPOp>(module->getOperation()), 0u);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }
}

TEST(OnlineAttentionDecompositionTest,
     CausalPositionsSurviveTailRefinementWithoutRedundantGuards) {
  for (llvm::StringRef dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(dtype.str() + " " + std::to_string(extent));
      auto context = createContext();
      auto module = parseOnlineModule(*context, extent, dtype, false);
      ASSERT_TRUE(module);
      auto region = findRegion(*module);
      LinalgExtOnlineAttentionOp source;
      region.walk([&](LinalgExtOnlineAttentionOp op) { source = op; });
      mlir::OpBuilder builder(source);
      auto zero =
          builder.create<mlir::arith::ConstantIndexOp>(source.getLoc(), 0);
      auto end =
          builder.create<mlir::arith::ConstantIndexOp>(source.getLoc(), extent);
      source.getPositionsMutable().append(mlir::ValueRange{zero, zero, end});
      source.setCausal(true);
      source.setZeroFullyMasked(true);
      source.setPositionMapAttr(mlir::AffineMapAttr::get(mlir::AffineMap::get(
          6, 0, {builder.getAffineDimExpr(2), builder.getAffineDimExpr(4)},
          context.get())));
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
      auto domain = buildTemporalDomain(region);
      ASSERT_TRUE(domain.succeeded());
      auto choice = selectK2Tile(*domain.domain, 128);
      StructuredMaterializationRelations relations;
      relations.structuralOutputs.push_back({0, region.getResult(0)});
      TemporalTilingFailure tilingFailure;
      ASSERT_TRUE(mlir::succeeded(applyTemporalTiling(
          {{*domain.domain, choice}}, relations, &tilingFailure)))
          << tilingFailure.detail;
      unsigned mainTiles = 0, tailTiles = 0;
      region.walk([&](LinalgExtOnlineAttentionOp op) {
        // This fixture starts with query row zero and spans all 1025 rows.
        // Every KV block is visible, but none is visible to the entire query
        // block. Both facts must remove the corresponding scalar branches.
        EXPECT_FALSE(op->getParentOfType<mlir::scf::IfOp>());
        EXPECT_TRUE(op.getCausal());
        EXPECT_EQ(op.getPositions().size(), 3u);
        EXPECT_EQ(op.getPositions()[2], end.getResult());
        if (auto loop = op->getParentOfType<mlir::scf::ForOp>()) {
          ++mainTiles;
          EXPECT_EQ(op.getPositions()[1], loop.getInductionVar());
        } else {
          ++tailTiles;
          EXPECT_EQ(mlir::getConstantIntValue(op.getPositions()[1]), 1024);
        }
      });
      EXPECT_EQ(mainTiles, 1u);
      EXPECT_EQ(tailTiles, extent == 1024 ? 0u : 1u);
      OnlineAttentionDecompositionFailure failure;
      auto decomposed = decomposeOnlineAttention(*module, relations, &failure);
      ASSERT_TRUE(mlir::succeeded(decomposed)) << failure.detail;
      EXPECT_EQ(countOps<LinalgExtOnlineAttentionOp>(module->getOperation()),
                0u);
      EXPECT_EQ(countOps<mlir::tensor::InsertOp>(module->getOperation()), 0u);
      unsigned validLengthTests = 0, causalTests = 0;
      region.walk([&](mlir::arith::CmpFOp cmp) {
        validLengthTests +=
            cmp.getPredicate() == mlir::arith::CmpFPredicate::OGE;
        causalTests += cmp.getPredicate() == mlir::arith::CmpFPredicate::OGT;
      });
      EXPECT_EQ(validLengthTests, 0u);
      EXPECT_EQ(causalTests, 0u);
      auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered = lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      auto movement = materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      ASSERT_TRUE(mlir::succeeded(lowerPhysicalToInstr(*module)));
      unsigned biasReads = 0;
      module->walk([&](InstrRDMAOp read) {
        auto type = mlir::cast<mlir::MemRefType>(read.getSource().getType());
        if (type.getElementType().isF32() && type.getRank() == 2) {
          ++biasReads;
          EXPECT_EQ(type.getDimSize(0), 1025);
          EXPECT_LE(type.getDimSize(1), 128);
          EXPECT_EQ(read.getByteCount(), type.getNumElements() * 4u);
        }
      });
      EXPECT_GT(biasReads, 0u);
    }
}

TEST(OnlineAttentionDecompositionTest, CausalLoopsVisitOnlyVisibleKeyBlocks) {
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t queryTile : {192, 256})
      for (int64_t queryOffset : {0, 37}) {
        SCOPED_TRACE(::testing::Message()
                     << extent << "/" << queryTile << "/" << queryOffset);
        auto context = createContext();
        auto module = parseOnlineModule(*context, extent, "bf16", false);
        ASSERT_TRUE(module);
        auto region = findRegion(*module);
        LinalgExtOnlineAttentionOp source;
        region.walk([&](LinalgExtOnlineAttentionOp op) { source = op; });
        mlir::OpBuilder builder(source);
        auto zero =
            builder.create<mlir::arith::ConstantIndexOp>(source.getLoc(), 0);
        auto queryStart = builder.create<mlir::arith::ConstantIndexOp>(
            source.getLoc(), queryOffset);
        auto end = builder.create<mlir::arith::ConstantIndexOp>(source.getLoc(),
                                                                extent);
        source.getPositionsMutable().append(
            mlir::ValueRange{queryStart, zero, end});
        source.setCausal(true);
        source.setPositionMapAttr(mlir::AffineMapAttr::get(mlir::AffineMap::get(
            6, 0, {builder.getAffineDimExpr(2), builder.getAffineDimExpr(4)},
            context.get())));
        auto domain = buildTemporalDomain(region);
        ASSERT_TRUE(domain.succeeded());
        auto choice = selectK2Tile(*domain.domain, 128, queryTile);
        StructuredMaterializationRelations relations;
        relations.structuralOutputs.push_back({0, region.getResult(0)});
        ASSERT_TRUE(mlir::succeeded(
            applyTemporalTiling({{*domain.domain, choice}}, relations)));
        llvm::DenseMap<mlir::Value, int64_t> values;
        std::function<int64_t(mlir::Value)> evaluate =
            [&](mlir::Value value) -> int64_t {
          if (auto constant = mlir::getConstantIntValue(value))
            return *constant;
          if (auto found = values.find(value); found != values.end())
            return found->second;
          if (auto add = value.getDefiningOp<mlir::arith::AddIOp>())
            return evaluate(add.getLhs()) + evaluate(add.getRhs());
          if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>())
            return evaluate(sub.getLhs()) - evaluate(sub.getRhs());
          if (auto minimum = value.getDefiningOp<mlir::arith::MinSIOp>())
            return std::min(evaluate(minimum.getLhs()),
                            evaluate(minimum.getRhs()));
          ADD_FAILURE() << "unexpected scalar in the actual KV loop bound";
          return -1;
        };
        unsigned boundedLoops = 0;
        region.walk([&](mlir::scf::ForOp loop) {
          if (mlir::getConstantIntValue(loop.getStep()) != 128)
            return;
          ++boundedLoops;
          auto outer = loop->getParentOfType<mlir::scf::ForOp>();
          if (!outer) {
            EXPECT_EQ(evaluate(loop.getUpperBound()), (extent / 128) * 128);
            return;
          }
          EXPECT_EQ(mlir::getConstantIntValue(outer.getStep()), queryTile);
          auto first = evaluate(outer.getLowerBound());
          auto last = evaluate(outer.getUpperBound());
          for (int64_t q = first; q < last; q += queryTile) {
            values[outer.getInductionVar()] = q;
            int64_t upper = evaluate(loop.getUpperBound());
            EXPECT_EQ(upper, std::min((extent / 128) * 128,
                                      q + queryOffset + queryTile));
            unsigned actual = 0, expected = 0;
            for (int64_t k = 0; k < upper; k += 128) {
              ++actual;
              EXPECT_LT(k, q + queryOffset + queryTile);
            }
            for (int64_t k = 0; k < (extent / 128) * 128; k += 128)
              expected += k < q + queryOffset + queryTile;
            EXPECT_EQ(actual, expected);
          }
        });
        EXPECT_GT(boundedLoops, 0u);
        ASSERT_TRUE(
            mlir::succeeded(decomposeOnlineAttention(*module, relations)));
        auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        auto lowered = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
        auto movement = materializeTileBoundaryMovement(*module, relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        auto placement = optimizePhysicalMovementPlacement(
            *module, relations, LayoutMaterializationPlacement::LoopInvariant);
        ASSERT_TRUE(mlir::succeeded(placement));
        EXPECT_TRUE(mlir::succeeded(lowerPhysicalToInstr(*module)));
      }
}

TEST(OnlineAttentionDecompositionTest,
     MainAndTailBecomeActualQKStateAndPVWithoutNewLoops) {
  for (llvm::StringRef dtype : {"f16", "bf16"})
    for (int64_t extent : {1024, 1025, 1031}) {
      SCOPED_TRACE(extent);
      std::unique_ptr<mlir::MLIRContext> context = createContext();
      auto module = parseOnlineModule(*context, extent, dtype,
                                      /*withMask=*/false);
      ASSERT_TRUE(module);
      TileRegionOp region = findRegion(*module);
      TemporalDomainResult domain = buildTemporalDomain(region);
      ASSERT_TRUE(domain.succeeded());
      TemporalChoice choice = selectK2Tile(*domain.domain, 128);
      StructuredMaterializationRelations relations;
      relations.structuralOutputs.push_back({0, region.getResult(0)});
      TemporalTilingFailure tilingFailure;
      auto tiled = applyTemporalTiling({{*domain.domain, choice}}, relations,
                                       &tilingFailure);
      ASSERT_TRUE(mlir::succeeded(tiled)) << tilingFailure.detail;
      const unsigned onlineBefore =
          countOps<LinalgExtOnlineAttentionOp>(module->getOperation());
      const unsigned loopsBefore =
          countOps<mlir::scf::ForOp>(module->getOperation());
      ASSERT_EQ(onlineBefore, extent == 1024 ? 1u : 2u);

      OnlineAttentionDecompositionFailure failure;
      auto decomposed = decomposeOnlineAttention(*module, relations, &failure);
      ASSERT_TRUE(mlir::succeeded(decomposed)) << failure.detail;
      EXPECT_EQ(decomposed->decomposedOperations, onlineBefore);
      EXPECT_EQ(decomposed->qkContractions, onlineBefore);
      EXPECT_EQ(decomposed->pvContractions, onlineBefore);
      EXPECT_EQ(decomposed->scoreApplications, onlineBefore);
      EXPECT_EQ(decomposed->rowReductions, onlineBefore * 2);
      EXPECT_EQ(decomposed->normalizationFactors, onlineBefore);
      EXPECT_EQ(decomposed->probabilityUpdates, onlineBefore);
      EXPECT_EQ(decomposed->stateScales, onlineBefore * 2);
      EXPECT_EQ(decomposed->scoreScratchTensors, onlineBefore);
      EXPECT_EQ(decomposed->maximumScoreElements, UINT64_C(2) * 4 * 1025 * 128);
      EXPECT_EQ(countOps<LinalgExtAttentionOp>(module->getOperation()), 0u);
      EXPECT_EQ(countOps<LinalgExtOnlineAttentionOp>(module->getOperation()),
                0u);
      EXPECT_EQ(countOps<mlir::scf::ForOp>(module->getOperation()),
                loopsBefore);
      EXPECT_EQ(countContractions(module->getOperation()), onlineBefore * 2);
      EXPECT_EQ(countRowReductions(module->getOperation()), onlineBefore * 2);
      EXPECT_EQ(countOps<mlir::math::ExpOp>(module->getOperation()),
                onlineBefore * 2);
      EXPECT_EQ(countOps<mlir::arith::CmpFOp>(module->getOperation()),
                onlineBefore);
      module->walk([&](mlir::arith::CmpFOp comparison) {
        auto rows = comparison->getParentOfType<mlir::linalg::GenericOp>();
        ASSERT_TRUE(rows);
        auto type =
            mlir::cast<mlir::RankedTensorType>(rows.getResult(0).getType());
        EXPECT_EQ(type.getShape(), llvm::ArrayRef<int64_t>({2, 4, 1025}));
        // Both old-state rescaling and score exponentiation consume this one
        // row value, while the original maximum still feeds the state yield.
        EXPECT_EQ(std::distance(rows.getResult(0).use_begin(),
                                rows.getResult(0).use_end()),
                  2);
      });
      module->walk([&](mlir::math::ExpOp exponential) {
        auto generic = exponential->getParentOfType<mlir::linalg::GenericOp>();
        EXPECT_EQ(countOps<mlir::arith::CmpFOp>(generic), 0u);
        EXPECT_EQ(countOps<mlir::arith::SelectOp>(generic), 0u);
        EXPECT_TRUE(
            exponential.getOperand().getDefiningOp<mlir::arith::SubFOp>());
      });
      module->walk([&](mlir::scf::ForOp loop) {
        EXPECT_EQ(loop.getNumRegionIterArgs(), 3u);
      });
      unsigned scoreScratch = 0;
      module->walk([&](mlir::linalg::GenericOp generic) {
        if (generic.getNumDpsInputs() != 2 || generic->getNumResults() != 1)
          return;
        mlir::Value query = generic.getDpsInputs()[0];
        while (auto slice = query.getDefiningOp<mlir::tensor::ExtractSliceOp>())
          query = slice.getSource();
        if (query != region.getBody().front().getArgument(0))
          return;
        auto type =
            mlir::cast<mlir::RankedTensorType>(generic.getResult(0).getType());
        ++scoreScratch;
        EXPECT_TRUE(type.getElementType().isF32());
        EXPECT_EQ(type.getShape()[0], 2);
        EXPECT_EQ(type.getShape()[1], 4);
        EXPECT_EQ(type.getShape()[2], 1025);
        EXPECT_LE(type.getShape()[3], 128);
      });
      EXPECT_EQ(scoreScratch, onlineBefore);
      EXPECT_TRUE(
          mlir::succeeded(verifyOnlineAttentionDecompositionComplete(*module)));
      EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
      LayoutOptimizationResult layout =
          resolveCurrentLayoutsAndBufferize(*module, relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      EXPECT_EQ(layout.statistics.bufferizationInvocations, 1u);
      unsigned carriedStates = 0;
      module->walk([&](mlir::scf::ForOp loop) {
        if (loop.getNumRegionIterArgs() != 3)
          return;
        for (auto argument : loop.getRegionIterArgs()) {
          auto type = mlir::cast<mlir::MemRefType>(argument.getType());
          EXPECT_EQ(getWaferMemoryAttr(type).getLayout(), MemLayout::NCx);
          ++carriedStates;
        }
      });
      EXPECT_EQ(carriedStates, 3u);
      EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
      EXPECT_TRUE(mlir::succeeded(verifyLayoutResolvedTileRegions(*module)));
      ASSERT_EQ(relations.structuralOutputs.size(), 1u);
      EXPECT_TRUE(isWaferDDRMemRefType(
          relations.structuralOutputs.front().endpoint.getType()));
      StructuredToTileResult tileLowering =
          lowerStructuredComputeToTile(*module, relations);
      ASSERT_TRUE(tileLowering.succeeded()) << tileLowering.detail;
      EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(module->getOperation()), 0u);
      // Probability narrows at the PV input, including the one-element tail.
      EXPECT_EQ(tileLowering.statistics.contractions, onlineBefore * 2);
      module->walk([&](ComputeGemmOp op) {
        auto lhs = mlir::cast<mlir::MemRefType>(op.getLhs().getType());
        if (lhs.getElementType().isF32()) {
          EXPECT_NE(lhs.getShape().back(), 1);
        }
      });
      EXPECT_EQ(tileLowering.statistics.reductions, onlineBefore * 2);
      EXPECT_TRUE(mlir::succeeded(verifyStructuredComputeLowered(*module)));
      EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
          module->getOperation(), relations)));
      BoundaryMovementResult movement =
          materializeTileBoundaryMovement(*module, relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_GT(movement.statistics.ddrLoads, 0u);
      EXPECT_EQ(movement.statistics.ddrStores, 1u);
      EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*module)));
      EXPECT_TRUE(mlir::succeeded(lowerPhysicalToInstr(*module)));
      EXPECT_LT(countOps<InstrGatherScatterOp>(module->getOperation()), 256u);
      EXPECT_LT(countOps<SyncNCCJoinOp>(module->getOperation()), 16u);
    }
}

TEST(OnlineAttentionDecompositionTest, ScoreRoundingSurvivesMainAndTail) {
  for (int64_t extent : {1024, 1025, 1031})
    for (llvm::StringRef storage : {"f16", "bf16"})
      for (bool withMask : {false, true}) {
        SCOPED_TRACE(std::to_string(extent) + ":" + storage.str() + ":" +
                     std::to_string(withMask));
        auto context = createContext();
        auto module = parseOnlineModule(*context, extent, storage, withMask);
        ASSERT_TRUE(module);
        LinalgExtOnlineAttentionOp online;
        module->walk([&](LinalgExtOnlineAttentionOp op) { online = op; });
        ASSERT_TRUE(online);
        auto storageType = online.getQuery().getType().getElementType();
        mlir::OpBuilder builder(online);
        auto loc = online.getLoc();
        auto scale = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getFloatAttr(storageType, 0.375));
        online.getScaleMutable().assign(scale.getResult());
        auto &body = online.getScoreRegion().front();
        while (!body.empty())
          body.back().erase();
        body.getArgument(1).setType(storageType);
        builder.setInsertionPointToStart(&body);
        auto dot = body.getArgument(0);
        auto scaleWide = builder.create<mlir::arith::ExtFOp>(
            loc, builder.getF32Type(), body.getArgument(1));
        auto scaled = builder.create<mlir::arith::MulFOp>(loc, dot, scaleWide);
        mlir::Value rounded =
            builder.create<mlir::arith::TruncFOp>(loc, storageType, scaled);
        if (withMask) {
          auto mask = builder.create<mlir::arith::TruncFOp>(
              loc, storageType, body.getArgument(2));
          rounded = builder.create<mlir::arith::AddFOp>(loc, rounded, mask);
        }
        auto widened = builder.create<mlir::arith::ExtFOp>(
            loc, builder.getF32Type(), rounded);
        builder.create<LinalgExtAttentionYieldOp>(loc, widened);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
        EXPECT_EQ(online.getScoreType(), builder.getF32Type());
        EXPECT_NE(online.getScoreType(), online.getScale().getType());
        auto signature = [](mlir::Block &block) {
          std::vector<std::pair<std::string, mlir::Type>> result;
          for (mlir::Operation &op : block.without_terminator())
            result.emplace_back(op.getName().getStringRef().str(),
                                op.getResult(0).getType());
          return result;
        };
        auto expected = signature(body);
        auto region = findRegion(*module);
        auto domain = buildTemporalDomain(region);
        ASSERT_TRUE(domain.succeeded());
        StructuredMaterializationRelations relations;
        relations.structuralOutputs.push_back({0, region.getResult(0)});
        auto choice = selectK2Tile(*domain.domain, 128);
        ASSERT_TRUE(mlir::succeeded(
            applyTemporalTiling({{*domain.domain, choice}}, relations)));
        unsigned updates = 0;
        module->walk([&](LinalgExtOnlineAttentionOp op) {
          EXPECT_EQ(signature(op.getScoreRegion().front()), expected);
          ++updates;
        });
        EXPECT_EQ(updates, extent == 1024 ? 1u : 2u);
        ASSERT_TRUE(
            mlir::succeeded(decomposeOnlineAttention(*module, relations)));
        unsigned scores = 0;
        module->walk([&](mlir::linalg::GenericOp op) {
          if (op.getNumDpsInputs() != (withMask ? 3 : 2) ||
              mlir::isa<mlir::ShapedType>(op.getDpsInputs()[1].getType()))
            return;
          EXPECT_EQ(signature(op.getRegion().front()), expected);
          ++scores;
        });
        EXPECT_EQ(scores, updates);
        auto layout = resolveCurrentLayoutsAndBufferize(*module, relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        auto lowered = lowerStructuredComputeToTile(*module, relations);
        ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
        auto movement = materializeTileBoundaryMovement(*module, relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        EXPECT_TRUE(mlir::succeeded(lowerPhysicalToInstr(*module)));
      }
}

TEST(OnlineAttentionDecompositionTest,
     BroadcastMaskAndBF16UseTheSameModeNeutralDecomposition) {
  for (auto [elementType, withMask] :
       {std::pair<llvm::StringRef, bool>{"f16", true}, {"bf16", false}}) {
    SCOPED_TRACE(elementType.str());
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    auto module = parseOnlineModule(*context, 1031, elementType, withMask);
    ASSERT_TRUE(module);
    TileRegionOp region = findRegion(*module);
    TemporalDomainResult domain = buildTemporalDomain(region);
    ASSERT_TRUE(domain.succeeded());
    TemporalChoice choice = selectK2Tile(*domain.domain, 128);
    StructuredMaterializationRelations relations;
    relations.structuralOutputs.push_back({0, region.getResult(0)});
    ASSERT_TRUE(mlir::succeeded(
        applyTemporalTiling({{*domain.domain, choice}}, relations)));
    const unsigned onlineBefore =
        countOps<LinalgExtOnlineAttentionOp>(module->getOperation());
    auto decomposed = decomposeOnlineAttention(*module, relations);
    ASSERT_TRUE(mlir::succeeded(decomposed));
    EXPECT_EQ(decomposed->decomposedOperations, onlineBefore);
    EXPECT_EQ(decomposed->scoreApplications, onlineBefore);
    unsigned maskConsumers = 0;
    module->walk([&](mlir::linalg::GenericOp generic) {
      if (llvm::any_of(generic.getDpsInputs(), [](mlir::Value value) {
            auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
            return type && type.getRank() == 3 && type.getShape()[1] == 1025;
          }))
        ++maskConsumers;
    });
    EXPECT_EQ(maskConsumers, withMask ? onlineBefore : 0u);
    EXPECT_EQ(countOps<LinalgExtOnlineAttentionOp>(module->getOperation()), 0u);
    LayoutOptimizationResult layout =
        resolveCurrentLayoutsAndBufferize(*module, relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    StructuredToTileResult tileLowering =
        lowerStructuredComputeToTile(*module, relations);
    ASSERT_TRUE(tileLowering.succeeded()) << tileLowering.detail;
    EXPECT_EQ(tileLowering.statistics.contractions, onlineBefore * 2);
    EXPECT_EQ(tileLowering.statistics.reductions, onlineBefore * 2);
    EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(module->getOperation()), 0u);
    BoundaryMovementResult movement =
        materializeTileBoundaryMovement(*module, relations);
    ASSERT_TRUE(movement.succeeded()) << movement.detail;
    EXPECT_GT(movement.statistics.ddrLoads, 0u);
    EXPECT_EQ(movement.statistics.ddrStores, 1u);
    EXPECT_TRUE(mlir::succeeded(lowerPhysicalToInstr(*module)));
    EXPECT_LT(countOps<InstrGatherScatterOp>(module->getOperation()), 256u);
    EXPECT_LT(countOps<SyncNCCJoinOp>(module->getOperation()), 16u);
  }
}

TEST(OnlineAttentionDecompositionTest, GraphAttentionFailsBeforeMutation) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  func.func @main(%query: tensor<2x1025x64xf16>,
                  %key: tensor<2x1031x64xf16>,
                  %value: tensor<2x1031x128xf16>, %scale: f32)
      -> tensor<2x1025x128xf16> {
    %empty = tensor.empty() : tensor<2x1025x128xf16>
    %result = wafer.linalg_ext.attention
        ins(%query, %key, %value, %scale : tensor<2x1025x64xf16>,
            tensor<2x1031x64xf16>, tensor<2x1031x128xf16>, f32)
        outs(%empty : tensor<2x1025x128xf16>)
        algorithm(<flash_attention>) indexing_maps = [#q, #k, #v, #s, #o] score {
    ^bb0(%attention_1_dot: f32, %attention_1_scale: f32):

      %attention_1_scaled = arith.mulf %attention_1_dot, %attention_1_scale : f32
      wafer.linalg_ext.attention.yield %attention_1_scaled : f32
    }
        -> tensor<2x1025x128xf16>
    return %result : tensor<2x1025x128xf16>
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  StructuredMaterializationRelations relations;
  OnlineAttentionDecompositionFailure failure;
  EXPECT_TRUE(
      mlir::failed(decomposeOnlineAttention(*module, relations, &failure)));
  EXPECT_EQ(failure.kind,
            OnlineAttentionDecompositionFailureKind::BrokenContract);
  LayoutOptimizationResult layout =
      resolveCurrentLayoutsAndBufferize(*module, relations);
  EXPECT_EQ(layout.status, ExactPBQPStatus::BrokenContract);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(OnlineAttentionDecompositionTest,
     DynamicOnlineTileReturnsUnsupportedWithoutMutation) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#acc = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#row = affine_map<(b, m, k1, k2, n) -> (b, m)>
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%query: tensor<2x1025x64xf16>,
                     %key: tensor<2x?x64xf16>,
                     %value: tensor<2x?x128xf16>)
        -> tensor<2x1025x128xf32> {
      %result = wafer.tile.region(
          %query, %key, %value : tensor<2x1025x64xf16>,
          tensor<2x?x64xf16>, tensor<2x?x128xf16>)
          -> (tensor<2x1025x128xf32>) {
      ^bb0(%query_arg: tensor<2x1025x64xf16>,
           %key_arg: tensor<2x?x64xf16>,
           %value_arg: tensor<2x?x128xf16>):
        %scale = arith.constant 1.0 : f32
        %accumulator = tensor.empty() : tensor<2x1025x128xf32>
        %maximum = tensor.empty() : tensor<2x1025xf32>
        %sum = tensor.empty() : tensor<2x1025xf32>
        %next_accumulator, %next_maximum, %next_sum =
            wafer.linalg_ext.online_attention
            ins(%query_arg, %key_arg, %value_arg, %scale :
                tensor<2x1025x64xf16>, tensor<2x?x64xf16>,
                tensor<2x?x128xf16>, f32)
            outs(%accumulator, %maximum, %sum : tensor<2x1025x128xf32>,
                tensor<2x1025xf32>, tensor<2x1025xf32>)
            indexing_maps = [#q, #k, #v, #s, #acc, #row, #row] score {
            ^bb0(%attention_2_dot: f32, %attention_2_scale: f32):

              %attention_2_scaled = arith.mulf %attention_2_dot, %attention_2_scale : f32
              wafer.linalg_ext.attention.yield %attention_2_scaled : f32
            }
            -> (tensor<2x1025x128xf32>, tensor<2x1025xf32>,
                tensor<2x1025xf32>)
        wafer.tile.yield %next_accumulator : tensor<2x1025x128xf32>
      }
      return %result : tensor<2x1025x128xf32>
    }
  }
}
)mlir",
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  TileRegionOp region = findRegion(*module);
  StructuredMaterializationRelations relations;
  relations.structuralOutputs.push_back({0, region.getResult(0)});
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  module->print(beforeStream);
  beforeStream.flush();
  OnlineAttentionDecompositionFailure failure;
  EXPECT_TRUE(
      mlir::failed(decomposeOnlineAttention(*module, relations, &failure)));
  EXPECT_EQ(failure.kind,
            OnlineAttentionDecompositionFailureKind::UnsupportedSemantics);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  module->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

} // namespace
