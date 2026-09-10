//===- SpatialRegionMaterializationTest.cpp ---------------------------===//

#include "Wafer/Transforms/Linalg/SpatialRegionMaterialization.h"
#include "Wafer/Transforms/Linalg/OnlineAttentionDecomposition.h"
#include "Wafer/Transforms/Linalg/StructuredTiling.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <set>
#include <type_traits>
#include <vector>

namespace {

std::unique_ptr<mlir::MLIRContext> createContext() {
  mlir::DialectRegistry registry;
  wafer::compiler::detail::registerCompilationDialects(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

llvm::SmallVector<wafer::TileId, 16> allTiles() {
  llvm::SmallVector<wafer::TileId, 16> result;
  for (int64_t tile = 0; tile < 16; ++tile)
    result.push_back(wafer::TileId(tile));
  return result;
}

std::string makeAttentionSource(llvm::StringRef algorithm) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%q: tensor<2x1025x128xf16>,
                  %k: tensor<2x1031x128xf16>,
                  %v: tensor<2x1031x64xf16>, %scale: f32)
      -> tensor<2x1025x64xf16> {
    %empty = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%empty : tensor<2x1025x64xf16>)
        algorithm(<)mlir"
         << algorithm << R"mlir(>) indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
)mlir";
  return text;
}

template <typename OpT> unsigned countOps(mlir::Operation *root) {
  unsigned count = 0;
  root->walk([&](OpT) { ++count; });
  return count;
}

wafer::TileModuleOp getTileOwner(mlir::Value value) {
  mlir::Operation *operation = nullptr;
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
    operation = result.getOwner();
  else if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    operation = argument.getOwner()->getParentOp();
  return operation ? operation->getParentOfType<wafer::TileModuleOp>()
                   : wafer::TileModuleOp{};
}

wafer::analysis::ExactIndexSet makeExactBox(llvm::ArrayRef<int64_t> offsets,
                                            llvm::ArrayRef<int64_t> sizes) {
  wafer::analysis::IndexSetResult set =
      wafer::analysis::IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(set.isExact());
  wafer::analysis::StaticRectangularIndexSet rectangle{llvm::to_vector(offsets),
                                                       llvm::to_vector(sizes)};
  return wafer::analysis::ExactIndexSet(
      std::move(*set.set), wafer::analysis::ExactIndexSetForm::BoxUnion,
      {rectangle});
}

mlir::FailureOr<wafer::compiler::detail::StructuredDAGAnalysis>
analyzeSingleTensorProgram(mlir::ModuleOp source, std::string &failureReason) {
  std::optional<wafer::compiler::detail::StructuredDAGAnalysis> selected;
  for (mlir::func::FuncOp function : source.getOps<mlir::func::FuncOp>()) {
    if (function.isExternal())
      continue;
    std::string candidateFailure;
    auto candidate = wafer::compiler::detail::StructuredDAGAnalysis::create(
        function, &candidateFailure);
    if (mlir::failed(candidate) || candidate->getNodes().empty())
      continue;
    if (selected) {
      failureReason = "test source contains multiple structured programs";
      return mlir::failure();
    }
    selected = std::move(*candidate);
  }
  if (!selected) {
    failureReason = "test source contains no structured TensorProgram";
    return mlir::failure();
  }
  return std::move(*selected);
}

mlir::FailureOr<wafer::SpatialRegionMaterializationResult>
materializeCanonical(mlir::ModuleOp source, std::string &failureReason) {
  using namespace wafer::compiler::detail;
  auto dag = analyzeSingleTensorProgram(source, failureReason);
  if (mlir::failed(dag))
    return mlir::failure();
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  if (mlir::failed(coordinate))
    return mlir::failure();
  auto demandSession = DemandPlanningSession::create(*dag, {}, &failureReason);
  if (mlir::failed(demandSession))
    return mlir::failure();
  wafer::analysis::ExactDemandOutcome demand =
      demandSession->query(coordinate->assignment);
  const wafer::analysis::ExactDemandProof *proof =
      wafer::analysis::getExactDemandProof(demand);
  if (!proof) {
    failureReason = std::visit(
        [](const auto &value) -> std::string {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, wafer::analysis::ExactDemandProof>)
            return {};
          else
            return value.detail;
        },
        demand);
    if (failureReason.empty())
      failureReason = "canonical demand did not produce an exact proof";
    return mlir::failure();
  }
  auto domain = RootWorkDomain::create(*dag, coordinate->assignment, *proof,
                                       allTiles(), &failureReason);
  if (mlir::failed(domain))
    return mlir::failure();
  RootWorkCollectionOutcome worksOutcome = collectRootWorks(*domain);
  RootWorkCollection *works = getRootWorkCollection(worksOutcome);
  if (!works) {
    failureReason = "canonical root-work collection failed";
    return mlir::failure();
  }
  CanonicalRegionPlanOutcome planOutcome =
      buildCanonicalRegionPlan(works->works);
  const RegionPlan *plan = getRegionPlan(planOutcome);
  if (!plan) {
    failureReason = "canonical RegionPlan construction failed";
    return mlir::failure();
  }
  llvm::SmallVector<StructuredOperationNodeMapping, 16> mappings;
  for (const StructuredDAGNode &node : dag->getNodes())
    mappings.push_back({node.operation, node.id});
  wafer::SpatialRegionMaterializationFailure failure;
  auto result =
      wafer::materializeSpatialRegions(source, wafer::CardId(0), allTiles(),
                                       mappings, works->works, *plan, &failure);
  if (mlir::failed(result))
    failureReason = failure.detail;
  return result;
}

mlir::FailureOr<wafer::SpatialRegionMaterializationResult>
materializeWithSpatialPlan(
    mlir::ModuleOp source,
    const std::function<void(wafer::compiler::detail::SpatialPlan &)> &select,
    std::string &failureReason) {
  using namespace wafer::compiler::detail;
  auto dag = analyzeSingleTensorProgram(source, failureReason);
  if (mlir::failed(dag))
    return mlir::failure();
  auto coordinate =
      buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
  if (mlir::failed(coordinate))
    return mlir::failure();
  SpatialPlan selected = coordinate->plan;
  select(selected);
  auto assignment =
      closeSpatialPlanStructure(coordinate->problem, selected, &failureReason);
  if (mlir::failed(assignment))
    return mlir::failure();
  auto demandSession = DemandPlanningSession::create(*dag, {}, &failureReason);
  if (mlir::failed(demandSession))
    return mlir::failure();
  wafer::analysis::ExactDemandOutcome demand =
      demandSession->query(*assignment);
  const wafer::analysis::ExactDemandProof *proof =
      wafer::analysis::getExactDemandProof(demand);
  if (!proof) {
    failureReason = "selected spatial plan has no exact demand proof";
    return mlir::failure();
  }
  auto domain = RootWorkDomain::create(*dag, *assignment, *proof, allTiles(),
                                       &failureReason);
  if (mlir::failed(domain))
    return mlir::failure();
  RootWorkCollectionOutcome worksOutcome = collectRootWorks(*domain);
  RootWorkCollection *works = getRootWorkCollection(worksOutcome);
  if (!works) {
    failureReason = "selected spatial plan has no root-work collection";
    return mlir::failure();
  }
  CanonicalRegionPlanOutcome planOutcome =
      buildCanonicalRegionPlan(works->works);
  const RegionPlan *plan = getRegionPlan(planOutcome);
  if (!plan) {
    failureReason = "selected spatial plan has no canonical RegionPlan";
    return mlir::failure();
  }
  llvm::SmallVector<StructuredOperationNodeMapping, 16> mappings;
  for (const StructuredDAGNode &node : dag->getNodes())
    mappings.push_back({node.operation, node.id});
  wafer::SpatialRegionMaterializationFailure failure;
  auto result =
      wafer::materializeSpatialRegions(source, wafer::CardId(0), allTiles(),
                                       mappings, works->works, *plan, &failure);
  if (mlir::failed(result))
    failureReason = failure.detail;
  return result;
}

mlir::FailureOr<wafer::SpatialRegionMaterializationResult>
materializeWithSpatialDomainPlan(
    mlir::ModuleOp source,
    const std::function<
        void(const wafer::compiler::detail::SpatialRootDomainFacts &,
             wafer::compiler::detail::SpatialPlan &, std::string &)> &select,
    wafer::SpatialRegionMaterializationFailure &materializationFailure,
    std::string &failureReason) {
  using namespace wafer::compiler::detail;
  auto dag = analyzeSingleTensorProgram(source, failureReason);
  if (mlir::failed(dag))
    return mlir::failure();
  auto topology = wafer::TargetTopology::create(source, &failureReason);
  if (mlir::failed(topology))
    return mlir::failure();
  auto spatial = buildSpatialPlanDomain(*dag, *topology, wafer::CardId(0));
  if (!spatial.succeeded()) {
    failureReason = "selected source has no SpatialDomain";
    return mlir::failure();
  }
  SpatialPlan plan = spatial.domain->getFirstPlan();
  const SpatialRootDomainFacts root =
      spatial.domain->getProblem().getRoots().front();
  select(root, plan, failureReason);
  if (!failureReason.empty())
    return mlir::failure();
  SpatialDomainEvaluation evaluation = spatial.domain->evaluate(*dag, plan);
  if (!evaluation.isSatisfied()) {
    failureReason = "selected SpatialDomain plan was not satisfied";
    return mlir::failure();
  }
  const wafer::analysis::ExactDemandProof *proof =
      wafer::analysis::getExactDemandProof(*evaluation.demand);
  if (!proof) {
    failureReason = "selected SpatialDomain plan has no exact demand proof";
    return mlir::failure();
  }
  auto domain = RootWorkDomain::create(*dag, *evaluation.assignment, *proof,
                                       allTiles(), &failureReason);
  if (mlir::failed(domain))
    return mlir::failure();
  RootWorkCollectionOutcome worksOutcome = collectRootWorks(*domain);
  RootWorkCollection *works = getRootWorkCollection(worksOutcome);
  if (!works) {
    failureReason = "selected SpatialDomain plan has no root work";
    return mlir::failure();
  }
  CanonicalRegionPlanOutcome planOutcome =
      buildCanonicalRegionPlan(works->works);
  const RegionPlan *regions = getRegionPlan(planOutcome);
  if (!regions) {
    failureReason = "selected SpatialDomain plan has no RegionPlan";
    return mlir::failure();
  }
  llvm::SmallVector<StructuredOperationNodeMapping, 16> mappings;
  for (const StructuredDAGNode &node : dag->getNodes())
    mappings.push_back({node.operation, node.id});
  auto result = wafer::materializeSpatialRegions(
      source, wafer::CardId(0), allTiles(), mappings, works->works, *regions,
      &materializationFailure);
  if (mlir::failed(result))
    failureReason = materializationFailure.detail;
  return result;
}

std::string makeIdentityGenericSource(llvm::ArrayRef<int64_t> shape) {
  std::string dimensions;
  std::string mapDimensions;
  std::string iteratorTypes;
  llvm::raw_string_ostream dimensionStream(dimensions);
  llvm::raw_string_ostream mapStream(mapDimensions);
  llvm::raw_string_ostream iteratorStream(iteratorTypes);
  for (auto [index, extent] : llvm::enumerate(shape)) {
    if (index) {
      mapStream << ", ";
      iteratorStream << ", ";
    }
    dimensionStream << extent << 'x';
    mapStream << 'd' << index;
    iteratorStream << "\"parallel\"";
  }
  dimensionStream.flush();
  mapStream.flush();
  iteratorStream.flush();
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "#id = affine_map<(" << mapDimensions << ") -> (" << mapDimensions
         << ")>\nmodule {\n"
         << "  wafer.target.topology @target {card_grid = array<i64: 1, 1>, "
            "card_interconnect = \"mesh\", tile_grid = array<i64: 4, 4>, "
            "unavailable_tiles = array<i64>}\n"
         << "  wafer.execution.mesh @logical {axes = [\"card\"], shape = "
            "array<i64: 1>}\n"
         << "  func.func @main(%input: tensor<" << dimensions
         << "f16>) -> tensor<" << dimensions << "f16> {\n"
         << "    %empty = tensor.empty() : tensor<" << dimensions << "f16>\n"
         << "    %result = linalg.generic {indexing_maps = [#id, #id], "
            "iterator_types = ["
         << iteratorTypes << "]}\n"
         << "        ins(%input : tensor<" << dimensions << "f16>)\n"
         << "        outs(%empty : tensor<" << dimensions << "f16>) {\n"
         << "      ^bb0(%value: f16, %old: f16):\n"
         << "        %next = arith.addf %value, %value : f16\n"
         << "        linalg.yield %next : f16\n"
         << "    } -> tensor<" << dimensions << "f16>\n"
         << "    return %result : tensor<" << dimensions << "f16>\n"
         << "  }\n}\n";
  stream.flush();
  return source;
}

TEST(SpatialRegionMaterializationTest,
     MaterializesAlignedAndRaggedStructuralRegions) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string sourceText;
    llvm::raw_string_ostream stream(sourceText);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @map(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> tensor<2x" << extent
           << R"mlir(x128xf16> {
    %empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>)
        outs(%empty : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x)mlir"
           << extent << R"mlir(x128xf16>
    return %result : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>
  }
}
)mlir";
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        sourceText, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
    std::string before;
    llvm::raw_string_ostream beforeStream(before);
    source->print(beforeStream);
    beforeStream.flush();

    std::string failureReason;
    auto materialized = materializeCanonical(*source, failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(
        countOps<wafer::TileModuleOp>(materialized->module->getOperation()),
        16u);
    EXPECT_EQ(
        countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
        16u);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(materialized->module->getOperation()),
              0u);
    EXPECT_EQ(
        countOps<mlir::memref::AllocOp>(materialized->module->getOperation()),
        0u);
    EXPECT_TRUE(materialized->relations.boundaryRelations.empty());
    EXPECT_EQ(materialized->relations.structuralOutputs.size(), 16u);
    EXPECT_EQ(
        countOps<mlir::linalg::GenericOp>(materialized->module->getOperation()),
        16u);
    EXPECT_TRUE(mlir::succeeded(
        wafer::verifyStructuralTileRegions(*materialized->module)));
    materialized->module->walk([&](wafer::TileRegionOp region) {
      EXPECT_TRUE(region->getParentOfType<wafer::TileModuleOp>());
      region.walk([&](mlir::linalg::LinalgOp op) {
        EXPECT_TRUE(mlir::isa<mlir::TilingInterface>(op.getOperation()));
      });
    });
    EXPECT_EQ(
        countOps<mlir::func::FuncOp>(materialized->module->getOperation()),
        16u);
    materialized->module->walk([&](wafer::TileModuleOp tile) {
      auto functions = tile.getOps<mlir::func::FuncOp>();
      ASSERT_NE(functions.begin(), functions.end());
      mlir::func::FuncOp onlyFunction = *functions.begin();
      EXPECT_EQ(onlyFunction.getSymName(), "entry");
      EXPECT_EQ(std::distance(functions.begin(), functions.end()), 1);
    });

    // Direct item-13 witness: the actual operation in the returned owner can
    // be consumed by the pinned TilingInterface helper without consulting the
    // Spatial/Region choice again. Use a clone because this test still checks
    // the unmodified item-12 relation set below.
    mlir::OwningOpRef<mlir::ModuleOp> downstream =
        materialized->module->clone();
    mlir::linalg::LinalgOp downstreamOp;
    downstream->walk([&](mlir::linalg::LinalgOp op) {
      if (!downstreamOp)
        downstreamOp = op;
    });
    ASSERT_TRUE(downstreamOp);
    mlir::OpBuilder downstreamBuilder(downstreamOp);
    downstreamBuilder.setInsertionPointAfter(downstreamOp);
    auto tiling =
        mlir::cast<mlir::TilingInterface>(downstreamOp.getOperation());
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
    llvm::SmallVector<mlir::OpFoldResult, 4> sizes;
    for (const mlir::Range &range :
         tiling.getIterationDomain(downstreamBuilder)) {
      offsets.push_back(range.offset);
      sizes.push_back(range.size);
    }
    std::string downstreamFailure;
    auto downstreamTile = wafer::materializeOperationFromIterationTile(
        downstreamOp, downstreamBuilder, offsets, sizes, &downstreamFailure);
    ASSERT_TRUE(mlir::succeeded(downstreamTile)) << downstreamFailure;
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*downstream)));

    std::string after;
    llvm::raw_string_ostream afterStream(after);
    source->print(afterStream);
    afterStream.flush();
    EXPECT_EQ(after, before);
  }
}

TEST(SpatialRegionMaterializationTest,
     MaterializesRankFourAndSixCurrentOperations) {
  const std::vector<std::vector<int64_t>> shapes = {{2, 4, 1025, 128},
                                                    {2, 2, 2, 2, 1031, 16}};
  for (const std::vector<int64_t> &shape : shapes) {
    SCOPED_TRACE(shape.size());
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        makeIdentityGenericSource(shape), mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    std::string failureReason;
    auto materialized = materializeCanonical(*source, failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(
        countOps<wafer::TileModuleOp>(materialized->module->getOperation()),
        16u);
    unsigned tileableOperations = 0;
    materialized->module->walk([&](mlir::TilingInterface tiling) {
      if (!tiling->getParentOfType<wafer::TileRegionOp>())
        return;
      ++tileableOperations;
      ASSERT_TRUE(tiling);
      EXPECT_EQ(tiling.getLoopIteratorTypes().size(), shape.size());
    });
    EXPECT_EQ(tileableOperations, 16u);
  }
}

TEST(SpatialRegionMaterializationTest,
     PreservesReferencedFunctionSymbolClosureWithoutScratchFunctions) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func private @twice(%value: f16) -> f16 {
    %result = arith.addf %value, %value : f16
    return %result : f16
  }
  func.func @main(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %empty = tensor.empty() : tensor<2x1025x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = func.call @twice(%value) : (f16) -> f16
        linalg.yield %next : f16
    } -> tensor<2x1025x128xf16>
    return %result : tensor<2x1025x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
  std::string failureReason;
  auto materialized = materializeCanonical(*source, failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<mlir::func::CallOp>(materialized->module->getOperation()),
            16u);
  EXPECT_EQ(countOps<mlir::func::FuncOp>(materialized->module->getOperation()),
            32u);
  unsigned topLevelHelpers = 0;
  for (mlir::func::FuncOp function :
       materialized->module->getOps<mlir::func::FuncOp>()) {
    ++topLevelHelpers;
    (void)function;
  }
  EXPECT_EQ(topLevelHelpers, 0u);
  materialized->module->walk([&](wafer::TileModuleOp tile) {
    std::set<llvm::StringRef> symbols;
    for (mlir::func::FuncOp function : tile.getOps<mlir::func::FuncOp>())
      symbols.insert(function.getSymName());
    EXPECT_EQ(symbols, (std::set<llvm::StringRef>{"entry", "twice"}));
  });
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
            16u);
}

TEST(SpatialRegionMaterializationTest,
     MaterializesOneTwoAndUniformTailSelectedWorkPieces) {
  for (unsigned selectedPieces : {1u, 2u}) {
    SCOPED_TRACE(selectedPieces);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        makeIdentityGenericSource({2, 1031, 128}),
        mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    std::string failureReason;
    auto materialized = materializeWithSpatialPlan(
        *source,
        [&](wafer::compiler::detail::SpatialPlan &plan) {
          auto &node = plan.nodes.front();
          node.axes = {
              {0,
               wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
               1},
              {1,
               wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
               static_cast<int64_t>(selectedPieces)},
              {2,
               wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
               1}};
          node.embedding.clear();
          for (unsigned tile = 0; tile < selectedPieces; ++tile)
            node.embedding.push_back(wafer::TileId(tile));
          node.reductionMerges.clear();
        },
        failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(
        countOps<wafer::TileModuleOp>(materialized->module->getOperation()),
        16u);
    EXPECT_EQ(
        countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
        selectedPieces);
    EXPECT_EQ(
        countOps<mlir::linalg::GenericOp>(materialized->module->getOperation()),
        selectedPieces);
  }

  std::unique_ptr<mlir::MLIRContext> context = createContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      makeIdentityGenericSource({2, 1031, 128}),
      mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string failureReason;
  auto materialized = materializeWithSpatialPlan(
      *source,
      [](wafer::compiler::detail::SpatialPlan &plan) {
        auto &node = plan.nodes.front();
        node.axes = {
            {0, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             1},
            {1, wafer::compiler::detail::IteratorPartitionScheme::UniformExtent,
             128},
            {2, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             1}};
        node.embedding.clear();
        for (int64_t tile = 0; tile < 9; ++tile)
          node.embedding.push_back(wafer::TileId(tile));
        node.reductionMerges.clear();
      },
      failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
            9u);
  unsigned fullPieces = 0;
  unsigned tailPieces = 0;
  materialized->module->walk([&](mlir::linalg::GenericOp operation) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(
        operation->getResult(0).getType());
    ASSERT_TRUE(type);
    if (type.getShape()[1] == 128)
      ++fullPieces;
    else if (type.getShape()[1] == 7)
      ++tailPieces;
  });
  EXPECT_EQ(fullPieces, 8u);
  EXPECT_EQ(tailPieces, 1u);
}

TEST(SpatialRegionMaterializationTest,
     MaterializesConnectedRegionWithoutImplicitProducerReplication) {
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string sourceText;
    llvm::raw_string_ostream stream(sourceText);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> tensor<2x" << extent
           << R"mlir(x128xf16> {
    %producer_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>)
        outs(%producer_empty : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x)mlir"
           << extent << R"mlir(x128xf16>
    %consumer_empty = tensor.empty() : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>
    %consumer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>)
        outs(%consumer_empty : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.mulf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x)mlir"
           << extent << R"mlir(x128xf16>
    return %consumer : tensor<2x)mlir"
           << extent << R"mlir(x128xf16>
  }
}
)mlir";
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        sourceText, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    mlir::func::FuncOp function = *source->getOps<mlir::func::FuncOp>().begin();
    std::string failureReason;
    auto dag = wafer::compiler::detail::StructuredDAGAnalysis::create(
        function, &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto coordinate = wafer::compiler::detail::buildCanonicalSpatialAssignment(
        *dag, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
    auto demandSession = wafer::compiler::detail::DemandPlanningSession::create(
        *dag, {}, &failureReason);
    ASSERT_TRUE(mlir::succeeded(demandSession)) << failureReason;
    auto demand = demandSession->query(coordinate->assignment);
    const auto *proof = wafer::analysis::getExactDemandProof(demand);
    ASSERT_NE(proof, nullptr);
    auto workDomain = wafer::compiler::detail::RootWorkDomain::create(
        *dag, coordinate->assignment, *proof, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(workDomain)) << failureReason;
    auto workOutcome = wafer::compiler::detail::collectRootWorks(*workDomain);
    auto *works = wafer::compiler::detail::getRootWorkCollection(workOutcome);
    ASSERT_NE(works, nullptr);
    auto canonicalPlanOutcome =
        wafer::compiler::detail::buildCanonicalRegionPlan(works->works);
    const auto *canonicalPlan =
        wafer::compiler::detail::getRegionPlan(canonicalPlanOutcome);
    ASSERT_NE(canonicalPlan, nullptr);
    llvm::SmallVector<wafer::compiler::detail::StructuredOperationNodeMapping,
                      16>
        mappings;
    for (const auto &node : dag->getNodes())
      mappings.push_back({node.operation, node.id});
    wafer::SpatialRegionMaterializationFailure canonicalFailure;
    auto canonical = wafer::materializeSpatialRegions(
        *source, wafer::CardId(0), allTiles(), mappings, works->works,
        *canonicalPlan, &canonicalFailure);
    ASSERT_TRUE(mlir::succeeded(canonical)) << canonicalFailure.detail;
    EXPECT_EQ(countOps<wafer::TileRegionOp>(canonical->module->getOperation()),
              32u);
    // Canonical elementwise placement keeps producer/consumer pieces on the
    // same Tile. Separate Regions are completely represented by direct SSA.
    EXPECT_TRUE(canonical->relations.boundaryRelations.empty());
    wafer::compiler::detail::RegionPlan reversed = *canonicalPlan;
    std::reverse(reversed.groups.begin(), reversed.groups.end());
    wafer::SpatialRegionMaterializationFailure reversedFailure;
    auto reversedMaterialization = wafer::materializeSpatialRegions(
        *source, wafer::CardId(0), allTiles(), mappings, works->works, reversed,
        &reversedFailure);
    ASSERT_TRUE(mlir::succeeded(reversedMaterialization))
        << reversedFailure.detail;
    std::string canonicalIR;
    std::string reversedIR;
    llvm::raw_string_ostream canonicalStream(canonicalIR);
    llvm::raw_string_ostream reversedStream(reversedIR);
    canonical->module->print(canonicalStream);
    reversedMaterialization->module->print(reversedStream);
    canonicalStream.flush();
    reversedStream.flush();
    EXPECT_EQ(reversedIR, canonicalIR);

    auto regionDomain = wafer::compiler::detail::RegionDomain::create(
        works->works, &failureReason);
    ASSERT_TRUE(mlir::succeeded(regionDomain)) << failureReason;
    std::optional<wafer::compiler::detail::RegionPlan> selected;
    std::optional<wafer::compiler::detail::RegionPlan> replicaPlan;
    for (const auto &proposal : regionDomain->getProposals(64))
      if (!selected && llvm::any_of(proposal.groups, [](const auto &group) {
            return group.mandatoryRoots.size() == 2 &&
                   !group.localBindings.empty();
          })) {
        selected = proposal;
      } else if (!replicaPlan &&
                 llvm::any_of(proposal.groups, [](const auto &group) {
                   return !group.replicas.empty();
                 }))
        replicaPlan = proposal;
    if (!replicaPlan) {
      auto successor = regionDomain->getFirstPlan();
      for (unsigned step = 0;
           step < 4096 &&
           successor.getKind() ==
               wafer::compiler::detail::RegionSuccessorKind::Plan;
           ++step) {
        const auto *candidate = successor.getPlan();
        if (candidate && llvm::any_of(candidate->groups, [](const auto &group) {
              return !group.replicas.empty();
            })) {
          replicaPlan = *candidate;
          break;
        }
        const auto *cursor = successor.getCursor();
        if (!cursor)
          break;
        successor = regionDomain->getNextPlan(*cursor);
      }
    }
    ASSERT_TRUE(selected.has_value());
    wafer::SpatialRegionMaterializationFailure failure;
    auto materialized = wafer::materializeSpatialRegions(
        *source, wafer::CardId(0), allTiles(), mappings, works->works,
        *selected, &failure);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failure.detail;
    EXPECT_EQ(
        countOps<wafer::TileModuleOp>(materialized->module->getOperation()),
        16u);
    EXPECT_EQ(
        countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
        selected->groups.size());
    EXPECT_EQ(
        countOps<mlir::linalg::GenericOp>(materialized->module->getOperation()),
        32u);
    EXPECT_TRUE(materialized->relations.boundaryRelations.empty());
    unsigned singletonRegions = 0;
    unsigned fusedRegions = 0;
    materialized->module->walk([&](wafer::TileRegionOp region) {
      unsigned linalgCount = 0;
      region.walk([&](mlir::linalg::LinalgOp) { ++linalgCount; });
      if (linalgCount == 1)
        ++singletonRegions;
      else if (linalgCount == 2)
        ++fusedRegions;
      else
        ADD_FAILURE() << "unexpected Linalg count in selected Region: "
                      << linalgCount;
    });
    EXPECT_EQ(singletonRegions, 30u);
    EXPECT_EQ(fusedRegions, 1u);
    ASSERT_TRUE(replicaPlan.has_value());
    wafer::SpatialRegionMaterializationFailure replicaFailure;
    auto replicated = wafer::materializeSpatialRegions(
        *source, wafer::CardId(0), allTiles(), mappings, works->works,
        *replicaPlan, &replicaFailure);
    ASSERT_TRUE(mlir::succeeded(replicated)) << replicaFailure.detail;
    size_t selectedOccurrences = 0;
    for (const auto &group : replicaPlan->groups)
      selectedOccurrences += group.executions.size() + group.replicas.size();
    EXPECT_EQ(
        countOps<mlir::linalg::GenericOp>(replicated->module->getOperation()),
        selectedOccurrences);
  }
}

TEST(SpatialRegionMaterializationTest,
     RecordsOnlyActualCrossTileBoundaryEndpoints) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %p0 = tensor.empty() : tensor<2x1025x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%p0 : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x128xf16>
    %c0 = tensor.empty() : tensor<2x1025x128xf16>
    %consumer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x1025x128xf16>)
        outs(%c0 : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.mulf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x128xf16>
    return %consumer : tensor<2x1025x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = *source->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = wafer::compiler::detail::StructuredDAGAnalysis::create(
      function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto coordinate = wafer::compiler::detail::buildCanonicalSpatialAssignment(
      *dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
  ASSERT_EQ(coordinate->plan.nodes.size(), 2u);
  // Keep the exact partition shapes but rotate the consumer embedding so the
  // same logical piece is supplied by a different physical Tile.
  std::rotate(coordinate->plan.nodes[1].embedding.begin(),
              std::next(coordinate->plan.nodes[1].embedding.begin()),
              coordinate->plan.nodes[1].embedding.end());
  auto shifted = wafer::compiler::detail::closeSpatialPlanStructure(
      coordinate->problem, coordinate->plan, &failureReason);
  ASSERT_TRUE(mlir::succeeded(shifted)) << failureReason;
  auto demandSession = wafer::compiler::detail::DemandPlanningSession::create(
      *dag, {}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(demandSession)) << failureReason;
  auto demand = demandSession->query(*shifted);
  const auto *proof = wafer::analysis::getExactDemandProof(demand);
  ASSERT_NE(proof, nullptr);
  auto workDomain = wafer::compiler::detail::RootWorkDomain::create(
      *dag, *shifted, *proof, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(workDomain)) << failureReason;
  auto workOutcome = wafer::compiler::detail::collectRootWorks(*workDomain);
  auto *works = wafer::compiler::detail::getRootWorkCollection(workOutcome);
  ASSERT_NE(works, nullptr);
  auto planOutcome =
      wafer::compiler::detail::buildCanonicalRegionPlan(works->works);
  const auto *plan = wafer::compiler::detail::getRegionPlan(planOutcome);
  ASSERT_NE(plan, nullptr);
  llvm::SmallVector<wafer::compiler::detail::StructuredOperationNodeMapping, 16>
      mappings;
  for (const auto &node : dag->getNodes())
    mappings.push_back({node.operation, node.id});
  wafer::SpatialRegionMaterializationFailure failure;
  auto materialized =
      wafer::materializeSpatialRegions(*source, wafer::CardId(0), allTiles(),
                                       mappings, works->works, *plan, &failure);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failure.detail;
  ASSERT_FALSE(materialized->relations.boundaryRelations.empty());
  for (const auto &relation : materialized->relations.boundaryRelations) {
    EXPECT_TRUE(relation.sourceEndpoint);
    EXPECT_TRUE(relation.destinationEndpoint);
    wafer::TileModuleOp sourceOwner = getTileOwner(relation.sourceEndpoint);
    wafer::TileModuleOp destinationOwner =
        getTileOwner(relation.destinationEndpoint);
    ASSERT_TRUE(sourceOwner);
    ASSERT_TRUE(destinationOwner);
    EXPECT_NE(sourceOwner, destinationOwner);
    EXPECT_EQ(relation.sourceEndpoint.getType(),
              relation.destinationEndpoint.getType());
  }
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          materialized->module->getOperation(), materialized->relations)));
}

TEST(SpatialRegionMaterializationTest,
     AssemblesExactMultiFragmentFaninAcrossUnequalPartitions) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @chain(%input: tensor<2x1031x128xf16>)
      -> tensor<2x1031x128xf16> {
    %producer_empty = tensor.empty() : tensor<2x1031x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1031x128xf16>)
        outs(%producer_empty : tensor<2x1031x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1031x128xf16>
    %consumer_empty = tensor.empty() : tensor<2x1031x128xf16>
    %consumer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x1031x128xf16>)
        outs(%consumer_empty : tensor<2x1031x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.mulf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1031x128xf16>
    return %consumer : tensor<2x1031x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string failureReason;
  auto materialized = materializeWithSpatialPlan(
      *source,
      [](wafer::compiler::detail::SpatialPlan &plan) {
        ASSERT_EQ(plan.nodes.size(), 2u);
        auto &producer = plan.nodes[0];
        producer.axes = {
            {0, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             1},
            {1, wafer::compiler::detail::IteratorPartitionScheme::UniformExtent,
             128},
            {2, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             1}};
        producer.embedding.clear();
        for (int64_t tile = 0; tile < 9; ++tile)
          producer.embedding.push_back(wafer::TileId(tile));
        auto &consumer = plan.nodes[1];
        consumer.axes = {
            {0, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             1},
            {1, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             8},
            {2, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             1}};
        consumer.embedding.clear();
        for (int64_t tile = 0; tile < 8; ++tile)
          consumer.embedding.push_back(wafer::TileId(tile));
      },
      failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(
      countOps<mlir::linalg::GenericOp>(materialized->module->getOperation()),
      17u);
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
            17u);
  EXPECT_EQ(materialized->relations.boundaryRelations.size(), 8u);
  std::set<std::pair<const void *, const void *>> endpoints;
  for (const auto &relation : materialized->relations.boundaryRelations) {
    EXPECT_TRUE(endpoints
                    .insert({relation.sourceEndpoint.getAsOpaquePointer(),
                             relation.destinationEndpoint.getAsOpaquePointer()})
                    .second);
    EXPECT_NE(getTileOwner(relation.sourceEndpoint),
              getTileOwner(relation.destinationEndpoint));
  }
  // Seventeen result-piece publications plus fourteen exact fragment inserts;
  // the count is proportional to selected pieces/fragments, never elements.
  EXPECT_EQ(countOps<mlir::tensor::InsertSliceOp>(
                materialized->module->getOperation()),
            31u);
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          materialized->module->getOperation(), materialized->relations)));
}

TEST(SpatialRegionMaterializationTest,
     MaterializesFlashAttentionAsOnlineState) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%q: tensor<2x1025x128xf16>,
                  %k: tensor<2x1031x128xf16>,
                  %v: tensor<2x1031x64xf16>, %scale: f32)
      -> tensor<2x1025x64xf16> {
    %empty = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%empty : tensor<2x1025x64xf16>)
        algorithm(<flash_attention>) indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string failureReason;
  auto materialized = materializeCanonical(*source, failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<wafer::LinalgExtAttentionOp>(
                materialized->module->getOperation()),
            0u);
  EXPECT_EQ(countOps<wafer::LinalgExtOnlineAttentionOp>(
                materialized->module->getOperation()),
            16u);
  EXPECT_EQ(
      countOps<mlir::linalg::MatmulOp>(materialized->module->getOperation()),
      0u);
  materialized->module->walk([&](wafer::LinalgExtOnlineAttentionOp attention) {
    EXPECT_EQ(attention.getNumResults(), 3u);
    auto keyType =
        mlir::cast<mlir::RankedTensorType>(attention.getKey().getType());
    EXPECT_EQ(keyType.getShape()[1], 1031);
  });
  EXPECT_EQ(countOps<wafer::LinalgExtAttentionOp>(source->getOperation()), 1u);
}

TEST(SpatialRegionMaterializationTest,
     MaterializedFAAndFDFeedTemporalTilingAndModeNeutralDecomposition) {
  for (llvm::StringRef algorithm : {llvm::StringRef("flash_attention"),
                                    llvm::StringRef("flash_decoding")}) {
    SCOPED_TRACE(algorithm.str());
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        makeAttentionSource(algorithm), mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    std::string failureReason;
    auto materialized = materializeCanonical(*source, failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    const unsigned onlineBefore = countOps<wafer::LinalgExtOnlineAttentionOp>(
        materialized->module->getOperation());
    ASSERT_GT(onlineBefore, 0u);
    const size_t boundaryCount =
        materialized->relations.boundaryRelations.size();
    const size_t outputCount = materialized->relations.structuralOutputs.size();
    llvm::SmallVector<std::pair<void *, void *>, 16> boundaryEndpoints;
    for (const auto &relation : materialized->relations.boundaryRelations)
      boundaryEndpoints.push_back(
          {relation.sourceEndpoint.getAsOpaquePointer(),
           relation.destinationEndpoint.getAsOpaquePointer()});
    llvm::SmallVector<void *, 16> outputEndpoints;
    for (const auto &relation : materialized->relations.structuralOutputs)
      outputEndpoints.push_back(relation.endpoint.getAsOpaquePointer());

    llvm::SmallVector<wafer::TileRegionOp, 32> regions;
    materialized->module->walk(
        [&](wafer::TileRegionOp region) { regions.push_back(region); });
    uint64_t tiledOnlineScopes = 0;
    uint64_t createdLoops = 0;
    for (wafer::TileRegionOp region : regions) {
      auto domain = wafer::compiler::detail::buildTemporalDomain(region);
      ASSERT_TRUE(domain.succeeded())
          << (domain.failure ? domain.failure->detail : "");
      auto first = domain.domain->getFirstChoice();
      ASSERT_EQ(first.getKind(),
                wafer::compiler::detail::TemporalSuccessorKind::Choice);
      ASSERT_NE(first.getChoice(), nullptr);
      wafer::compiler::detail::TemporalChoice choice = *first.getChoice();
      llvm::ArrayRef<wafer::compiler::detail::TemporalScopeDescriptor>
          descriptors = domain.domain->getScopeDescriptors();
      bool selectedOnline = false;
      for (auto [scope, descriptor] :
           llvm::zip_equal(choice.scopes, descriptors)) {
        auto online = mlir::dyn_cast<wafer::LinalgExtOnlineAttentionOp>(
            descriptor.operation);
        if (!online)
          continue;
        auto roles = online.getIterationRoles();
        ASSERT_TRUE(mlir::succeeded(roles));
        for (unsigned dimension : roles->keyValueReduction) {
          const int64_t extent = descriptor.iterationExtents[dimension];
          if (extent <= 1)
            continue;
          scope.iteratorTileSizes[dimension] = std::min<int64_t>(32, extent);
          selectedOnline |= scope.iteratorTileSizes[dimension] < extent;
        }
        auto order = wafer::compiler::detail::buildFirstTemporalLoopOrder(
            descriptor.iterationExtents, scope.iteratorTileSizes,
            descriptor.precedence);
        ASSERT_TRUE(mlir::succeeded(order));
        scope.loopOrder = std::move(*order);
        ++tiledOnlineScopes;
      }
      if (!selectedOnline)
        continue;
      ASSERT_TRUE(domain.domain->contains(choice));
      wafer::TemporalTilingFailure failure;
      auto tiled = wafer::applyTemporalTiling(
          *domain.domain, choice, materialized->relations, &failure);
      ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
      createdLoops += tiled->loops;
    }

    EXPECT_EQ(tiledOnlineScopes, onlineBefore);
    EXPECT_GE(createdLoops, onlineBefore);
    EXPECT_EQ(countOps<wafer::LinalgExtAttentionOp>(
                  materialized->module->getOperation()),
              0u);
    EXPECT_GE(countOps<wafer::LinalgExtOnlineAttentionOp>(
                  materialized->module->getOperation()),
              onlineBefore);
    EXPECT_EQ(
        countOps<mlir::linalg::MatmulOp>(materialized->module->getOperation()),
        0u);
    unsigned onlineLoops = 0;
    materialized->module->walk([&](mlir::scf::ForOp loop) {
      if (loop.getBody()->getOps<wafer::LinalgExtOnlineAttentionOp>().empty())
        return;
      ++onlineLoops;
      EXPECT_EQ(loop.getNumRegionIterArgs(), 3u);
    });
    EXPECT_GE(onlineLoops, onlineBefore);
    EXPECT_EQ(materialized->relations.boundaryRelations.size(), boundaryCount);
    EXPECT_EQ(materialized->relations.structuralOutputs.size(), outputCount);
    for (auto [index, relation] :
         llvm::enumerate(materialized->relations.boundaryRelations))
      EXPECT_EQ(
          std::make_pair(relation.sourceEndpoint.getAsOpaquePointer(),
                         relation.destinationEndpoint.getAsOpaquePointer()),
          boundaryEndpoints[index]);
    for (auto [index, relation] :
         llvm::enumerate(materialized->relations.structuralOutputs))
      EXPECT_EQ(relation.endpoint.getAsOpaquePointer(), outputEndpoints[index]);
    EXPECT_TRUE(mlir::succeeded(
        wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
            materialized->module->getOperation(), materialized->relations)));
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized->module)));

    const unsigned onlineAfterTiling =
        countOps<wafer::LinalgExtOnlineAttentionOp>(
            materialized->module->getOperation());
    const unsigned loopsBeforeDecomposition =
        countOps<mlir::scf::ForOp>(materialized->module->getOperation());
    wafer::OnlineAttentionDecompositionFailure decompositionFailure;
    auto decomposed = wafer::decomposeOnlineAttention(
        *materialized->module, materialized->relations, &decompositionFailure);
    ASSERT_TRUE(mlir::succeeded(decomposed)) << decompositionFailure.detail;
    EXPECT_EQ(decomposed->decomposedOperations, onlineAfterTiling);
    EXPECT_EQ(decomposed->qkContractions, onlineAfterTiling);
    EXPECT_EQ(decomposed->pvContractions, onlineAfterTiling);
    EXPECT_EQ(countOps<wafer::LinalgExtAttentionOp>(
                  materialized->module->getOperation()),
              0u);
    EXPECT_EQ(countOps<wafer::LinalgExtOnlineAttentionOp>(
                  materialized->module->getOperation()),
              0u);
    EXPECT_EQ(countOps<mlir::scf::ForOp>(materialized->module->getOperation()),
              loopsBeforeDecomposition);
    EXPECT_EQ(materialized->relations.boundaryRelations.size(), boundaryCount);
    EXPECT_EQ(materialized->relations.structuralOutputs.size(), outputCount);
    for (auto [index, relation] :
         llvm::enumerate(materialized->relations.boundaryRelations))
      EXPECT_EQ(
          std::make_pair(relation.sourceEndpoint.getAsOpaquePointer(),
                         relation.destinationEndpoint.getAsOpaquePointer()),
          boundaryEndpoints[index]);
    for (auto [index, relation] :
         llvm::enumerate(materialized->relations.structuralOutputs))
      EXPECT_EQ(relation.endpoint.getAsOpaquePointer(), outputEndpoints[index]);
    EXPECT_TRUE(
        mlir::succeeded(wafer::verifyOnlineAttentionDecompositionComplete(
            *materialized->module)));
    EXPECT_TRUE(mlir::succeeded(
        wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
            materialized->module->getOperation(), materialized->relations)));
    wafer::compiler::detail::LayoutOptimizationResult layout =
        wafer::compiler::detail::resolveCurrentLayoutsAndBufferize(
            *materialized->module, materialized->relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    EXPECT_EQ(layout.statistics.bufferizationInvocations, 1u);
    EXPECT_EQ(layout.statistics.redundantPublicationCopies, 0u);
    EXPECT_TRUE(mlir::succeeded(
        wafer::compiler::detail::verifyLayoutResolvedTileRegions(
            *materialized->module)));
    EXPECT_EQ(materialized->relations.boundaryRelations.size(), boundaryCount);
    EXPECT_EQ(materialized->relations.structuralOutputs.size(), outputCount);
    for (const auto &relation : materialized->relations.structuralOutputs)
      EXPECT_TRUE(wafer::isWaferDDRMemRefType(relation.endpoint.getType()));
  }
}

TEST(SpatialRegionMaterializationTest,
     MaterializesReductionContributionsAndActualMerges) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#input = affine_map<(b, n, k) -> (b, n, k)>
#output = affine_map<(b, n, k) -> (b, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x128x1025xf16>,
                  %init: tensor<2x128xf16>) -> tensor<2x128xf16> {
    %result = linalg.generic {
        indexing_maps = [#input, #output],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%input : tensor<2x128x1025xf16>)
        outs(%init : tensor<2x128xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<2x128xf16>
    return %result : tensor<2x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  mlir::func::FuncOp function = *source->getOps<mlir::func::FuncOp>().begin();
  std::string failureReason;
  auto dag = wafer::compiler::detail::StructuredDAGAnalysis::create(
      function, &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto topology = wafer::TargetTopology::create(*source, &failureReason);
  ASSERT_TRUE(mlir::succeeded(topology)) << failureReason;
  auto spatial = wafer::compiler::detail::buildSpatialPlanDomain(
      *dag, *topology, wafer::CardId(0));
  ASSERT_TRUE(spatial.succeeded());
  wafer::compiler::detail::SpatialRootDomainFacts root =
      spatial.domain->getProblem().getRoots().front();
  auto plan = spatial.domain->getFirstPlan();
  auto &node = plan.nodes.front();
  node.axes = {
      {0, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts, 2},
      {1, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts, 1},
      {2, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts, 2}};
  node.embedding = {wafer::TileId(0), wafer::TileId(1), wafer::TileId(4),
                    wafer::TileId(5)};
  auto groups = wafer::compiler::detail::deriveSpatialReductionGroups(
      root, node.axes, &failureReason);
  ASSERT_TRUE(mlir::succeeded(groups)) << failureReason;
  ASSERT_EQ(groups->size(), 2u);
  node.reductionMerges = {{(*groups)[0], wafer::TileId(15)},
                          {(*groups)[1], wafer::TileId(0)}};
  auto evaluation = spatial.domain->evaluate(*dag, plan);
  ASSERT_TRUE(evaluation.isSatisfied());
  const auto *proof = wafer::analysis::getExactDemandProof(*evaluation.demand);
  ASSERT_NE(proof, nullptr);
  auto workDomain = wafer::compiler::detail::RootWorkDomain::create(
      *dag, *evaluation.assignment, *proof, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(workDomain)) << failureReason;
  auto workOutcome = wafer::compiler::detail::collectRootWorks(*workDomain);
  auto *works = wafer::compiler::detail::getRootWorkCollection(workOutcome);
  ASSERT_NE(works, nullptr);
  auto regionPlanOutcome =
      wafer::compiler::detail::buildCanonicalRegionPlan(works->works);
  const auto *regionPlan =
      wafer::compiler::detail::getRegionPlan(regionPlanOutcome);
  ASSERT_NE(regionPlan, nullptr);
  llvm::SmallVector<wafer::compiler::detail::StructuredOperationNodeMapping, 16>
      mappings;
  for (const auto &dagNode : dag->getNodes())
    mappings.push_back({dagNode.operation, dagNode.id});
  wafer::SpatialRegionMaterializationFailure failure;
  auto materialized = wafer::materializeSpatialRegions(
      *source, wafer::CardId(0), allTiles(), mappings, works->works,
      *regionPlan, &failure);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failure.detail;
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
            regionPlan->groups.size());
  unsigned emptyShells = 0;
  materialized->module->walk([&](wafer::TileRegionOp region) {
    bool hasCompute = false;
    region.walk([&](mlir::linalg::LinalgOp) { hasCompute = true; });
    if (!hasCompute)
      ++emptyShells;
  });
  EXPECT_EQ(emptyShells, 0u);
  EXPECT_EQ(materialized->relations.structuralOutputs.size(), 2u);
  EXPECT_FALSE(materialized->relations.boundaryRelations.empty());
  EXPECT_EQ(
      countOps<mlir::linalg::ReduceOp>(materialized->module->getOperation()),
      2u);
  unsigned initRegionOperands = 0;
  materialized->module->walk([&](wafer::TileRegionOp region) {
    for (mlir::Value operand : region.getInputs())
      if (operand.getType() ==
          mlir::RankedTensorType::get({2, 128},
                                      mlir::Float16Type::get(context.get())))
        ++initRegionOperands;
  });
  EXPECT_EQ(initRegionOperands, 2u);
}

TEST(SpatialRegionMaterializationTest,
     MaterializesAlignedAndRaggedContractionPieces) {
  for (int64_t reductionExtent : {1024, 1025}) {
    SCOPED_TRACE(reductionExtent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string sourceText;
    llvm::raw_string_ostream stream(sourceText);
    stream << R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%lhs: tensor<2x128x)mlir"
           << reductionExtent << "xf16>, %rhs: tensor<2x" << reductionExtent
           << R"mlir(x64xf16>) -> tensor<2x128x64xf16> {
    %zero = arith.constant 0.0 : f16
    %empty = tensor.empty() : tensor<2x128x64xf16>
    %init = linalg.fill ins(%zero : f16)
        outs(%empty : tensor<2x128x64xf16>) -> tensor<2x128x64xf16>
    %result = linalg.batch_matmul
        ins(%lhs, %rhs : tensor<2x128x)mlir"
           << reductionExtent << "xf16>, tensor<2x" << reductionExtent
           << R"mlir(x64xf16>)
        outs(%init : tensor<2x128x64xf16>) -> tensor<2x128x64xf16>
    return %result : tensor<2x128x64xf16>
  }
}
)mlir";
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        sourceText, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    std::string failureReason;
    auto materialized = materializeCanonical(*source, failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(countOps<mlir::linalg::BatchMatmulOp>(
                  materialized->module->getOperation()),
              16u);
    materialized->module->walk([&](mlir::linalg::BatchMatmulOp contraction) {
      auto lhsType = mlir::cast<mlir::RankedTensorType>(
          contraction.getInputs()[0].getType());
      EXPECT_EQ(lhsType.getShape()[2], reductionExtent);
    });
  }
}

TEST(SpatialRegionMaterializationTest,
     MaterializesNonTrailingReductionAndPreservesSource) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#input = affine_map<(b, k, n) -> (b, k, n)>
#output = affine_map<(b, k, n) -> (b, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x1025x128xf16>,
                  %init: tensor<2x128xf16>) -> tensor<2x128xf16> {
    %result = linalg.generic {
        indexing_maps = [#input, #output],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%init : tensor<2x128xf16>) {
      ^bb0(%value: f16, %acc: f16):
        %next = arith.addf %value, %acc : f16
        linalg.yield %next : f16
    } -> tensor<2x128xf16>
    return %result : tensor<2x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  source->print(beforeStream);
  beforeStream.flush();
  std::string failureReason;
  wafer::SpatialRegionMaterializationFailure materializationFailure;
  auto materialized = materializeWithSpatialDomainPlan(
      *source,
      [](const wafer::compiler::detail::SpatialRootDomainFacts &root,
         wafer::compiler::detail::SpatialPlan &plan,
         std::string &failureReason) {
        auto &node = plan.nodes.front();
        node.axes = {
            {0, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             2},
            {1, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             2},
            {2, wafer::compiler::detail::IteratorPartitionScheme::BalancedParts,
             1}};
        node.embedding = {wafer::TileId(0), wafer::TileId(1), wafer::TileId(4),
                          wafer::TileId(5)};
        auto groups = wafer::compiler::detail::deriveSpatialReductionGroups(
            root, node.axes, &failureReason);
        if (mlir::failed(groups))
          return;
        node.reductionMerges.clear();
        for (auto [index, group] : llvm::enumerate(*groups))
          node.reductionMerges.push_back(
              {group, index == 0 ? wafer::TileId(15) : wafer::TileId(0)});
      },
      materializationFailure, failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(materializationFailure.kind,
            wafer::SpatialRegionMaterializationFailureKind::None);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*materialized->module)));
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  source->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(SpatialRegionMaterializationTest,
     MaterializesFlashDecodingOnlineContributionsAndActualMerge) {
  for (int64_t keyValueExtent : {1024, 1025, 1031}) {
    SCOPED_TRACE(keyValueExtent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string sourceText;
    llvm::raw_string_ostream stream(sourceText);
    stream << R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @decode(%q: tensor<2x1025x128xf16>,
                    %k: tensor<2x)mlir"
           << keyValueExtent << "x128xf16>, %v: tensor<2x" << keyValueExtent
           << R"mlir(x64xf16>, %scale: f32)
      -> tensor<2x1025x64xf16> {
    %empty = tensor.empty() : tensor<2x1025x64xf16>
    %result = wafer.linalg_ext.attention
        ins(%q, %k, %v, %scale : tensor<2x1025x128xf16>, tensor<2x)mlir"
           << keyValueExtent << "x128xf16>, tensor<2x" << keyValueExtent
           << R"mlir(x64xf16>, f32)
        outs(%empty : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>) indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    return %result : tensor<2x1025x64xf16>
  }
}
)mlir";
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        sourceText, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    std::string failureReason;
    auto materialized = materializeCanonical(*source, failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    unsigned mergeExecutions = materialized->relations.structuralOutputs.size();
    EXPECT_GE(mergeExecutions, 1u);
    EXPECT_EQ(countOps<wafer::LinalgExtAttentionOp>(
                  materialized->module->getOperation()),
              0u);
    const unsigned onlineCount = countOps<wafer::LinalgExtOnlineAttentionOp>(
        materialized->module->getOperation());
    EXPECT_GE(onlineCount, mergeExecutions * 2);
    materialized->module->walk(
        [&](wafer::LinalgExtOnlineAttentionOp attention) {
          auto keyType =
              mlir::cast<mlir::RankedTensorType>(attention.getKey().getType());
          auto valueType = mlir::cast<mlir::RankedTensorType>(
              attention.getValue().getType());
          EXPECT_GT(keyType.getShape()[1], 0);
          EXPECT_LT(keyType.getShape()[1], keyValueExtent);
          EXPECT_EQ(valueType.getShape()[1], keyType.getShape()[1]);
          EXPECT_EQ(attention.getNumResults(), 3u);
        });
    unsigned emptyShells = 0;
    materialized->module->walk([&](wafer::TileRegionOp region) {
      bool hasWork = false;
      region.walk([&](mlir::Operation *operation) {
        if (mlir::isa<wafer::LinalgExtOnlineAttentionOp,
                      mlir::linalg::GenericOp>(operation))
          hasWork = true;
      });
      if (!hasWork)
        ++emptyShells;
    });
    EXPECT_EQ(emptyShells, 0u);
    EXPECT_EQ(materialized->relations.structuralOutputs.size(),
              mergeExecutions);
    EXPECT_GE(materialized->relations.boundaryRelations.size(),
              mergeExecutions * 3);
    EXPECT_EQ(
        countOps<mlir::linalg::MatmulOp>(materialized->module->getOperation()),
        0u);
  }
}

TEST(SpatialRegionMaterializationTest,
     RejectsFAAndFDMismatchBeforeCandidateMutation) {
  using namespace wafer::compiler::detail;
  for (const auto &[plannedName, mutatedAlgorithm] :
       {std::pair<llvm::StringRef, wafer::AttentionAlgorithm>{
            "flash_attention", wafer::AttentionAlgorithm::FlashDecoding},
        {"flash_decoding", wafer::AttentionAlgorithm::FlashAttention}}) {
    SCOPED_TRACE(plannedName.str());
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        makeAttentionSource(plannedName), mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    std::string failureReason;
    auto dag = analyzeSingleTensorProgram(*source, failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto coordinate =
        buildCanonicalSpatialAssignment(*dag, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
    auto demandSession =
        DemandPlanningSession::create(*dag, {}, &failureReason);
    ASSERT_TRUE(mlir::succeeded(demandSession)) << failureReason;
    auto demand = demandSession->query(coordinate->assignment);
    const auto *proof = wafer::analysis::getExactDemandProof(demand);
    ASSERT_NE(proof, nullptr);
    auto workDomain = RootWorkDomain::create(
        *dag, coordinate->assignment, *proof, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(workDomain)) << failureReason;
    auto workOutcome = collectRootWorks(*workDomain);
    auto *works = getRootWorkCollection(workOutcome);
    ASSERT_NE(works, nullptr);
    auto planOutcome = buildCanonicalRegionPlan(works->works);
    const auto *plan = getRegionPlan(planOutcome);
    ASSERT_NE(plan, nullptr);
    llvm::SmallVector<StructuredOperationNodeMapping, 16> mappings;
    for (const auto &node : dag->getNodes())
      mappings.push_back({node.operation, node.id});

    wafer::LinalgExtAttentionOp attention;
    source->walk([&](wafer::LinalgExtAttentionOp op) { attention = op; });
    ASSERT_TRUE(attention);
    attention.setAlgorithm(mutatedAlgorithm);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
    std::string before;
    llvm::raw_string_ostream beforeStream(before);
    source->print(beforeStream);
    beforeStream.flush();
    wafer::SpatialRegionMaterializationFailure failure;
    auto materialized = wafer::materializeSpatialRegions(
        *source, wafer::CardId(0), allTiles(), mappings, works->works, *plan,
        &failure);
    EXPECT_TRUE(mlir::failed(materialized));
    EXPECT_EQ(failure.kind,
              wafer::SpatialRegionMaterializationFailureKind::BrokenContract);
    std::string after;
    llvm::raw_string_ostream afterStream(after);
    source->print(afterStream);
    afterStream.flush();
    EXPECT_EQ(after, before);
  }
}

TEST(SpatialRegionMaterializationTest,
     FlashDecodingActualMergeFeedsDownstreamSSA) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @decode(%qv: tensor<2x1025x128xf16>,
                    %kv: tensor<2x1031x128xf16>,
                    %vv: tensor<2x1031x64xf16>, %scale: f32)
      -> tensor<2x1025x64xf16> {
    %attention_empty = tensor.empty() : tensor<2x1025x64xf16>
    %attention = wafer.linalg_ext.attention
        ins(%qv, %kv, %vv, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%attention_empty : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>) indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    %consumer_empty = tensor.empty() : tensor<2x1025x64xf16>
    %consumer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%attention : tensor<2x1025x64xf16>)
        outs(%consumer_empty : tensor<2x1025x64xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x64xf16>
    return %consumer : tensor<2x1025x64xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string failureReason;
  auto materialized = materializeCanonical(*source, failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(countOps<wafer::LinalgExtAttentionOp>(
                materialized->module->getOperation()),
            0u);
  EXPECT_GE(countOps<wafer::LinalgExtOnlineAttentionOp>(
                materialized->module->getOperation()),
            16u);
  EXPECT_EQ(materialized->relations.structuralOutputs.size(), 16u);
  EXPECT_GT(
      countOps<mlir::linalg::GenericOp>(materialized->module->getOperation()),
      16u);
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          materialized->module->getOperation(), materialized->relations)));
}

TEST(SpatialRegionMaterializationTest,
     FlashDecodingOnlineContributionsConsumeStructuredProducerEndpoints) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
#q = affine_map<(b, m, k1, k2, n) -> (b, m, k1)>
#k = affine_map<(b, m, k1, k2, n) -> (b, k2, k1)>
#v = affine_map<(b, m, k1, k2, n) -> (b, k2, n)>
#s = affine_map<(b, m, k1, k2, n) -> ()>
#o = affine_map<(b, m, k1, k2, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @decode(%query_input: tensor<2x1025x128xf16>,
                    %key: tensor<2x1031x128xf16>,
                    %value: tensor<2x1031x64xf16>, %scale: f32)
      -> tensor<2x1025x64xf16> {
    %query_empty = tensor.empty() : tensor<2x1025x128xf16>
    %query = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%query_input : tensor<2x1025x128xf16>)
        outs(%query_empty : tensor<2x1025x128xf16>) {
      ^bb0(%input: f16, %old: f16):
        %projected = arith.addf %input, %input : f16
        linalg.yield %projected : f16
    } -> tensor<2x1025x128xf16>
    %expanded = tensor.expand_shape %query [[0], [1], [2, 3]]
        output_shape [2, 1025, 8, 16]
        : tensor<2x1025x128xf16> into tensor<2x1025x8x16xf16>
    %reshaped = tensor.collapse_shape %expanded [[0], [1], [2, 3]]
        : tensor<2x1025x8x16xf16> into tensor<2x1025x128xf16>
    %attention_empty = tensor.empty() : tensor<2x1025x64xf16>
    %attention = wafer.linalg_ext.attention
        ins(%reshaped, %key, %value, %scale : tensor<2x1025x128xf16>,
            tensor<2x1031x128xf16>, tensor<2x1031x64xf16>, f32)
        outs(%attention_empty : tensor<2x1025x64xf16>)
        algorithm(<flash_decoding>) indexing_maps = [#q, #k, #v, #s, #o]
        -> tensor<2x1025x64xf16>
    return %attention : tensor<2x1025x64xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string failureReason;
  auto materialized = materializeCanonical(*source, failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_GT(materialized->relations.boundaryRelations.size(), 0u);
  for (const auto &relation : materialized->relations.boundaryRelations) {
    EXPECT_TRUE(relation.sourceEndpoint);
    EXPECT_TRUE(relation.destinationEndpoint);
  }
  EXPECT_EQ(materialized->relations.structuralOutputs.size(), 8u);
  EXPECT_EQ(countOps<wafer::LinalgExtAttentionOp>(
                materialized->module->getOperation()),
            0u);
  const unsigned onlineCount = countOps<wafer::LinalgExtOnlineAttentionOp>(
      materialized->module->getOperation());
  EXPECT_GE(onlineCount, 16u);
  EXPECT_EQ(countOps<mlir::tensor::ExpandShapeOp>(
                materialized->module->getOperation()),
            onlineCount);
  EXPECT_EQ(countOps<mlir::tensor::CollapseShapeOp>(
                materialized->module->getOperation()),
            onlineCount);
  EXPECT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          materialized->module->getOperation(), materialized->relations)));
}

TEST(SpatialRegionMaterializationTest,
     FanoutKeepsOneProducerOccurrencePerSelectedTile) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @fanout(%input: tensor<2x1025x128xf16>)
      -> (tensor<2x1025x128xf16>, tensor<2x1025x128xf16>) {
    %p0 = tensor.empty() : tensor<2x1025x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%p0 : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x128xf16>
    %l0 = tensor.empty() : tensor<2x1025x128xf16>
    %left = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x1025x128xf16>)
        outs(%l0 : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.mulf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x128xf16>
    %r0 = tensor.empty() : tensor<2x1025x128xf16>
    %right = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%producer : tensor<2x1025x128xf16>)
        outs(%r0 : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.subf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x128xf16>
    return %left, %right : tensor<2x1025x128xf16>,
                           tensor<2x1025x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string failureReason;
  auto materialized = materializeCanonical(*source, failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(
      countOps<mlir::linalg::GenericOp>(materialized->module->getOperation()),
      48u);
  EXPECT_EQ(countOps<mlir::arith::AddFOp>(materialized->module->getOperation()),
            16u);
  EXPECT_TRUE(materialized->relations.boundaryRelations.empty());
  EXPECT_EQ(materialized->relations.structuralOutputs.size(), 32u);
}

TEST(SpatialRegionMaterializationTest,
     FifteenConsumersAndJointFaninRemainLinearInSelectedExecutions) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral type = "tensor<2x1025x128xf16>";
  std::string sourceText;
  llvm::raw_string_ostream stream(sourceText);
  stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @joint(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    %producer_empty = tensor.empty() : tensor<2x1025x128xf16>
    %producer = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%producer_empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x128xf16>
)mlir";
  for (unsigned consumer = 0; consumer < 15; ++consumer)
    stream << "    %empty" << consumer << " = tensor.empty() : " << type
           << "\n    %leaf" << consumer
           << " = linalg.generic {indexing_maps = [#id, #id], "
              "iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n"
           << "        ins(%producer : " << type << ") outs(%empty" << consumer
           << " : " << type << ") {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        %next = arith.mulf %value, %value : f16\n"
           << "        linalg.yield %next : f16\n"
           << "    } -> " << type << "\n";
  stream << "    %join_empty = tensor.empty() : " << type
         << "\n    %joined = linalg.generic {indexing_maps = [";
  for (unsigned index = 0; index < 16; ++index) {
    if (index)
      stream << ", ";
    stream << "#id";
  }
  stream << "], iterator_types = [\"parallel\", \"parallel\", "
            "\"parallel\"]}\n        ins(";
  for (unsigned index = 0; index < 15; ++index) {
    if (index)
      stream << ", ";
    stream << "%leaf" << index;
  }
  stream << " : ";
  for (unsigned index = 0; index < 15; ++index) {
    if (index)
      stream << ", ";
    stream << type;
  }
  stream << ") outs(%join_empty : " << type << ") {\n      ^bb0(";
  for (unsigned index = 0; index < 15; ++index) {
    if (index)
      stream << ", ";
    stream << "%value" << index << ": f16";
  }
  stream << ", %old: f16):\n";
  for (unsigned index = 1; index < 15; ++index)
    stream << "        %sum" << index << " = arith.addf "
           << (index == 1 ? "%value0" : ("%sum" + std::to_string(index - 1)))
           << ", %value" << index << " : f16\n";
  stream << "        linalg.yield %sum14 : f16\n    } -> " << type
         << "\n    return %joined : " << type << "\n  }\n}\n";
  stream.flush();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string failureReason;
  auto materialized = materializeCanonical(*source, failureReason);
  ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
  EXPECT_EQ(
      countOps<mlir::linalg::GenericOp>(materialized->module->getOperation()),
      272u);
  EXPECT_EQ(countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
            272u);
  EXPECT_EQ(materialized->relations.structuralOutputs.size(), 16u);
  EXPECT_TRUE(materialized->relations.boundaryRelations.empty());
}

TEST(SpatialRegionMaterializationTest,
     RejectsMalformedPlanWithoutMutatingSource) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
module {
  func.func @main(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1025x128xf16> {
    return %input : tensor<2x1025x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  source->print(beforeStream);
  beforeStream.flush();
  wafer::compiler::detail::RegionPlan emptyPlan;
  wafer::SpatialRegionMaterializationFailure failure;
  auto result = wafer::materializeSpatialRegions(
      *source, wafer::CardId(0), allTiles(), /*operationNodes=*/{},
      /*rootWorks=*/{}, emptyPlan, &failure);
  EXPECT_TRUE(mlir::failed(result));
  EXPECT_EQ(failure.kind,
            wafer::SpatialRegionMaterializationFailureKind::BrokenContract);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  source->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(SpatialRegionMaterializationTest,
     RejectsResidualObservableSupportWithoutMutatingSource) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x1025x128xf16>)
      -> tensor<2050x128xf16> {
    %empty = tensor.empty() : tensor<2x1025x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%empty : tensor<2x1025x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x1025x128xf16>
    %collapsed = tensor.collapse_shape %result [[0, 1], [2]]
        : tensor<2x1025x128xf16> into tensor<2050x128xf16>
    return %collapsed : tensor<2050x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  source->print(beforeStream);
  beforeStream.flush();
  std::string failureReason;
  auto materialized = materializeCanonical(*source, failureReason);
  EXPECT_TRUE(mlir::failed(materialized));
  EXPECT_NE(failureReason.find("direct structured result"), std::string::npos)
      << failureReason;
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  source->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(SpatialRegionMaterializationTest,
     PostMutationUnsupportedDestroysCandidateAndPreservesSource) {
  using namespace wafer::analysis;
  using namespace wafer::compiler::detail;
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x?x128xf16>, %extent: index)
      -> tensor<2x?x128xf16> {
    %empty = tensor.empty(%extent) : tensor<2x?x128xf16>
    %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x?x128xf16>)
        outs(%empty : tensor<2x?x128xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %value : f16
        linalg.yield %next : f16
    } -> tensor<2x?x128xf16>
    return %result : tensor<2x?x128xf16>
  }
}
)mlir";
  auto source = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(source);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
  mlir::func::FuncOp function = *source->getOps<mlir::func::FuncOp>().begin();
  mlir::Operation *rootOperation = nullptr;
  function.walk([&](mlir::linalg::GenericOp generic) {
    rootOperation = generic.getOperation();
  });
  ASSERT_NE(rootOperation, nullptr);

  SemanticRootKey root;
  root.anchorIndex = 0;
  LogicalShardId shard{root, {0, 0, 0}};
  ExactIndexSet domain = makeExactBox({0, 0, 0}, {2, 1025, 128});
  RootRegionWork work;
  work.id = {root, wafer::TileId(0)};
  work.rootOperation = rootOperation;
  work.execution.push_back({shard, {{0, 2}, {0, 1025}, {0, 128}}});
  RootUseId use{0, shard};
  RootOperandWork operand;
  operand.operand = 0;
  operand.uses.push_back({use, domain, domain});
  work.operands.push_back(std::move(operand));
  RootBoundaryWork boundary;
  boundary.id = {RootBoundaryKind::ProgramInput, root, 0};
  boundary.sourceValue = function.getArgument(0);
  boundary.requiredDomain = domain;
  boundary.consumerUses.push_back({use, domain, {}});
  work.boundaries.push_back(std::move(boundary));
  work.results.push_back({0, shard, std::nullopt, domain});
  CanonicalRegionPlanOutcome planOutcome = buildCanonicalRegionPlan({work});
  const RegionPlan *plan = getRegionPlan(planOutcome);
  ASSERT_NE(plan, nullptr);

  std::string before;
  llvm::raw_string_ostream beforeStream(before);
  source->print(beforeStream);
  beforeStream.flush();
  wafer::SpatialRegionMaterializationFailure failure;
  llvm::SmallVector<StructuredOperationNodeMapping, 1> mappings{
      {rootOperation, 0}};
  llvm::SmallVector<RootRegionWork, 1> works{work};
  auto materialized = wafer::materializeSpatialRegions(
      *source, wafer::CardId(0), allTiles(), mappings, works, *plan, &failure);
  EXPECT_TRUE(mlir::failed(materialized));
  EXPECT_EQ(failure.kind,
            wafer::SpatialRegionMaterializationFailureKind::Unsupported);
  std::string after;
  llvm::raw_string_ostream afterStream(after);
  source->print(afterStream);
  afterStream.flush();
  EXPECT_EQ(after, before);
}

TEST(SpatialRegionMaterializationTest,
     StageVerifiersDistinguishStructuralAndPhysicalBoundaries) {
  std::unique_ptr<mlir::MLIRContext> context = createContext();
  constexpr llvm::StringLiteral sourceText = R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(
        %arg0: memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>) {
      %unused = wafer.tile.region(
          %arg0 : memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>)
          -> (memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>) {
      ^bb0(%boundary: memref<2x1025x128xf16,
                               #wafer.memory<ddr, tensor>>):
        wafer.tile.yield %boundary
            : memref<2x1025x128xf16, #wafer.memory<ddr, tensor>>
      }
      return
    }
  }
}
)mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      sourceText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(module);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  mlir::ScopedDiagnosticHandler suppress(
      context.get(), [](mlir::Diagnostic &) { return mlir::success(); });
  EXPECT_TRUE(mlir::failed(wafer::verifyStructuralTileRegions(*module)));
  EXPECT_TRUE(
      mlir::succeeded(wafer::verifyTileRegionStorageBoundaries(*module)));

  constexpr llvm::StringLiteral structuralText = R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%arg0: tensor<2x1025x128xf16>) {
      %unused = wafer.tile.region(
          %arg0 : tensor<2x1025x128xf16>)
          -> (tensor<2x1025x128xf16>) {
      ^bb0(%boundary: tensor<2x1025x128xf16>):
        wafer.tile.yield %boundary : tensor<2x1025x128xf16>
      }
      return
    }
  }
}
)mlir";
  auto structural = mlir::parseSourceString<mlir::ModuleOp>(
      structuralText, mlir::ParserConfig(context.get()));
  ASSERT_TRUE(structural);
  EXPECT_TRUE(mlir::succeeded(wafer::verifyStructuralTileRegions(*structural)));
  EXPECT_TRUE(
      mlir::failed(wafer::verifyTileRegionStorageBoundaries(*structural)));
}

TEST(SpatialRegionMaterializationTest,
     MaterializesNonProjectedResultMapAsOneUnpartitionedRegion) {
  // A result indexing map that is not a projected permutation keeps the root in
  // the spatial domain as one unpartitioned cell, so its Tile IR must come out
  // of the ordinary materialization path: one TileRegion on one Tile and no
  // cross-Tile boundary relation. The pairs cover 1024 and the non-divisible
  // 1025.
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    std::unique_ptr<mlir::MLIRContext> context = createContext();
    std::string sourceText;
    llvm::raw_string_ostream stream(sourceText);
    stream << R"mlir(
module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @map(%input: tensor<2x)mlir"
           << extent << R"mlir(x4x4xf16>, %init: tensor<2x)mlir"
           << extent << R"mlir(x7xf16>) -> tensor<2x)mlir"
           << extent << R"mlir(x7xf16> {
    %result = linalg.generic {
        indexing_maps = [affine_map<(b, m, n, k) -> (b, m, n, k)>,
                         affine_map<(b, m, n, k) -> (b, m, n + k)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x)mlir"
           << extent << R"mlir(x4x4xf16>)
        outs(%init : tensor<2x)mlir"
           << extent << R"mlir(x7xf16>) {
      ^bb0(%value: f16, %old: f16):
        %next = arith.addf %value, %old : f16
        linalg.yield %next : f16
    } -> tensor<2x)mlir"
           << extent << R"mlir(x7xf16>
    return %result : tensor<2x)mlir"
           << extent << R"mlir(x7xf16>
  }
}
)mlir";
    auto source = mlir::parseSourceString<mlir::ModuleOp>(
        sourceText, mlir::ParserConfig(context.get()));
    ASSERT_TRUE(source);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
    std::string before;
    llvm::raw_string_ostream beforeStream(before);
    source->print(beforeStream);
    beforeStream.flush();

    std::string failureReason;
    auto materialized = materializeCanonical(*source, failureReason);
    ASSERT_TRUE(mlir::succeeded(materialized)) << failureReason;
    EXPECT_EQ(
        countOps<wafer::TileModuleOp>(materialized->module->getOperation()),
        16u);
    EXPECT_EQ(
        countOps<wafer::TileRegionOp>(materialized->module->getOperation()),
        1u);
    EXPECT_TRUE(materialized->relations.boundaryRelations.empty());
    EXPECT_EQ(materialized->relations.structuralOutputs.size(), 1u);
    std::string after;
    llvm::raw_string_ostream afterStream(after);
    source->print(afterStream);
    afterStream.flush();
    EXPECT_EQ(after, before);
  }
}

} // namespace
