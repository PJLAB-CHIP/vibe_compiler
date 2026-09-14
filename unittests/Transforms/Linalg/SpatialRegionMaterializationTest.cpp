//===- SpatialRegionMaterializationTest.cpp ---------------------------===//

#include "Wafer/Transforms/Linalg/SpatialRegionMaterialization.h"
#include "TestSupport/CodeGen/ExecutableTestSupport.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialPartitionPropagation.h"
#include "Wafer/Transforms/Linalg/OnlineAttentionDecomposition.h"
#include "Wafer/Transforms/Linalg/StructuredTiling.h"
#include "Wafer/Transforms/Linalg/TemporalTiling.h"
#include "Wafer/Transforms/Tile/LayoutOptimization.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "Wafer/Driver/CompilationInternal.h"
#include "Wafer/Driver/CurrentIRExecutablePipeline.h"
#include "Wafer/Driver/PhysicalDataflow/SearchCurrentIR.h"
#include "Wafer/IR/WaferDialect.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRegionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RootWorkDomain.h"
#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"
#include "Wafer/Planning/PhysicalDataflow/StructuredDemandAnalysis.h"
#include "Wafer/Transforms/Tile/BoundaryMovement.h"
#include "Wafer/Transforms/Tile/StructuredToTile.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
         << algorithm << R"mlir(>) indexing_maps = [#q, #k, #v, #s, #o] score {
    ^bb0(%attention_0_dot: f16, %attention_0_scale: f32):
      %attention_0_converted = arith.extf %attention_0_dot : f16 to f32
      %attention_0_scaled = arith.mulf %attention_0_converted, %attention_0_scale : f32
      wafer.linalg_ext.attention.yield %attention_0_scaled : f32
    }
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
      if (!selected && proposal.groups.size() == works->works.size() - 1 &&
          llvm::any_of(proposal.groups, [](const auto &group) {
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
    auto incomplete = *replicaPlan;
    auto replicaGroup = llvm::find_if(incomplete.groups, [](const auto &group) {
      return !group.replicas.empty();
    });
    ASSERT_NE(replicaGroup, incomplete.groups.end());
    ASSERT_FALSE(replicaGroup->replicas.front().inputs.empty());
    replicaGroup->replicas.front().inputs.pop_back();
    wafer::SpatialRegionMaterializationFailure missingInput;
    EXPECT_TRUE(mlir::failed(wafer::materializeSpatialRegions(
        *source, wafer::CardId(0), allTiles(), mappings, works->works,
        incomplete, &missingInput)));
    EXPECT_EQ(missingInput.kind,
              wafer::SpatialRegionMaterializationFailureKind::BrokenContract);
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
     CoalescesReshapedColumnDemandWithoutMergingSpatialOwners) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  for (int64_t extent : {1024, 1025, 1031}) {
    SCOPED_TRACE(extent);
    auto context = createContext();
    std::string text;
    llvm::raw_string_ostream out(text);
    std::string sourceType = "tensor<2x" + std::to_string(extent) + "x128xf16>";
    std::string expandedType =
        "tensor<2x" + std::to_string(extent) + "x4x32xf16>";
    std::string outputType =
        "tensor<2x4x" + std::to_string(extent) + "x32xf16>";
    out << R"mlir(module {
  wafer.target.topology @target {card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>,
      unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: )mlir"
        << sourceType << ") -> " << outputType << " {\n"
        << "%empty = tensor.empty() : " << sourceType << "\n"
        << "%producer = linalg.map ins(%input : " << sourceType
        << ") outs(%empty : " << sourceType << R"mlir() (%x: f16) {
    %twice = arith.addf %x, %x : f16
    linalg.yield %twice : f16
  }
  %expanded = tensor.expand_shape %producer [[0], [1], [2, 3]]
      output_shape [2, )mlir"
        << extent << ", 4, 32] : " << sourceType << " into " << expandedType
        << "\n%out = tensor.empty() : " << outputType << R"mlir(
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(b, h, m, n) -> (b, m, h, n)>,
                       affine_map<(b, h, m, n) -> (b, h, m, n)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : )mlir"
        << expandedType << ") outs(%out : " << outputType << R"mlir() {
    ^bb0(%x: f16, %old: f16):
      %square = arith.mulf %x, %x : f16
      linalg.yield %square : f16
  } -> )mlir"
        << outputType << "\nreturn %consumer : " << outputType << "\n}}\n";
    auto source = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    ASSERT_TRUE(source);
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
    std::string detail;
    auto actual = materializeWithSpatialPlan(
        *source,
        [](SpatialPlan &plan) {
          ASSERT_EQ(plan.nodes.size(), 2u);
          for (auto &node : plan.nodes) {
            for (auto &axis : node.axes) {
              axis.scheme = IteratorPartitionScheme::BalancedParts;
              axis.parameter = 1;
            }
            node.axes[1].parameter = 4;
            node.embedding = {TileId(0), TileId(1), TileId(2), TileId(3)};
            node.reductionMerges.clear();
          }
        },
        detail);
    ASSERT_TRUE(mlir::succeeded(actual)) << detail;
    ASSERT_TRUE(mlir::succeeded(mlir::verify(*actual->module)));
    // The producer splits rows; each consumer selects a different 32-column
    // slab through expand_shape. Four independent owners supply each slab.
    // Count only source-coordinate assemblies, excluding result publication.
    unsigned copies = 0;
    std::vector<unsigned> coverage(4 * extent);
    actual->module->walk([&](mlir::tensor::InsertSliceOp insert) {
      auto slice =
          insert.getSource().getDefiningOp<mlir::tensor::ExtractSliceOp>();
      if (!slice || slice.getSource().getType() != insert.getDest().getType() ||
          insert.getDestType().getShape() !=
              (llvm::ArrayRef<int64_t>{2, extent, 128}))
        return;
      ++copies;
      auto offsets = slice.getStaticOffsets();
      auto sizes = slice.getStaticSizes();
      EXPECT_EQ(offsets, insert.getStaticOffsets());
      EXPECT_EQ(sizes, insert.getStaticSizes());
      EXPECT_EQ(offsets[0], 0);
      EXPECT_EQ(sizes[0], 2);
      EXPECT_EQ(sizes[2], 32);
      ASSERT_GE(offsets[2], 0);
      ASSERT_LT(offsets[2], 128);
      EXPECT_EQ(offsets[2] % 32, 0);
      ASSERT_GE(offsets[1], 0);
      ASSERT_LE(offsets[1] + sizes[1], extent);
      EXPECT_GE(sizes[1], extent / 4);
      for (int64_t row = offsets[1]; row < offsets[1] + sizes[1]; ++row)
        ++coverage[(offsets[2] / 32) * extent + row];
    });
    EXPECT_EQ(copies, 16u);
    EXPECT_TRUE(
        llvm::all_of(coverage, [](unsigned count) { return count == 1; }));
    EXPECT_EQ(countOps<TileRegionOp>(actual->module->getOperation()), 8u);
    EXPECT_FALSE(actual->relations.boundaryRelations.empty());
    EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
        actual->module->getOperation(), actual->relations)));
    auto layout =
        resolveCurrentLayoutsAndBufferize(*actual->module, actual->relations);
    ASSERT_TRUE(layout.succeeded()) << layout.detail;
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*actual->module)));
  }
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
        algorithm(<flash_attention>) indexing_maps = [#q, #k, #v, #s, #o] score {
    ^bb0(%attention_1_dot: f16, %attention_1_scale: f32):
      %attention_1_converted = arith.extf %attention_1_dot : f16 to f32
      %attention_1_scaled = arith.mulf %attention_1_converted, %attention_1_scale : f32
      wafer.linalg_ext.attention.yield %attention_1_scaled : f32
    }
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
          {{*domain.domain, choice}}, materialized->relations, &failure);
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
      0u);
  unsigned localReductions = 0;
  materialized->module->walk([&](mlir::linalg::LinalgOp op) {
    if (op.getNumReductionLoops()) {
      ++localReductions;
      EXPECT_EQ(mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
                    .getRank(),
                2);
    }
  });
  EXPECT_EQ(localReductions, 4u);
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
     PartialMergePreservesPermutedOutputCoordinates) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (int64_t parts : {4, 16}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(parts);
      auto context = createContext();
      std::string text;
      llvm::raw_string_ostream out(text);
      out << R"mlir(module {
  wafer.target.topology @target {card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>,
      unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<2x)mlir"
          << extent
          << R"mlir(x3xf16>, %init: tensor<3x2xf16>) -> tensor<3x2xf16> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(b, k, n) -> (b, k, n)>,
                       affine_map<(b, k, n) -> (n, b)>],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%input : tensor<2x)mlir"
          << extent << R"mlir(x3xf16>) outs(%init : tensor<3x2xf16>) {
      ^bb0(%x: f16, %old: f16):
        %sum = arith.addf %x, %old : f16
        linalg.yield %sum : f16
    } -> tensor<3x2xf16>
    return %result : tensor<3x2xf16>
  }
})mlir";
      auto source =
          mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
      ASSERT_TRUE(source);
      wafer::SpatialRegionMaterializationFailure failure;
      std::string detail;
      auto actual = materializeWithSpatialDomainPlan(
          *source,
          [&](const wafer::compiler::detail::SpatialRootDomainFacts &root,
              wafer::compiler::detail::SpatialPlan &plan, std::string &reason) {
            auto &node = plan.nodes.front();
            for (auto &axis : node.axes) {
              axis.scheme = wafer::compiler::detail::IteratorPartitionScheme::
                  BalancedParts;
              axis.parameter = axis.iterator == 1 ? parts : 1;
            }
            node.embedding.clear();
            for (int64_t tile = 0; tile < parts; ++tile)
              node.embedding.push_back(wafer::TileId(tile));
            auto groups = wafer::compiler::detail::deriveSpatialReductionGroups(
                root, node.axes, &reason);
            if (mlir::failed(groups))
              return;
            node.reductionMerges.clear();
            for (const auto &group : *groups)
              node.reductionMerges.push_back({group, wafer::TileId(15)});
          },
          failure, detail);
      ASSERT_TRUE(mlir::succeeded(actual)) << detail;
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*actual->module)));
      std::vector<unsigned> coverage(extent);
      unsigned contributions = 0;
      actual->module->walk([&](mlir::linalg::LinalgOp op) {
        if (!op.getNumReductionLoops())
          return;
        ++contributions;
        EXPECT_EQ(op.getNumReductionLoops(), 1u);
        EXPECT_EQ(mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
                      .getShape(),
                  (llvm::ArrayRef<int64_t>{3, 2}));
        mlir::Value input = op.getDpsInputs()[0];
        auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
        int64_t offset = 0;
        while (true) {
          if (auto slice =
                  input.getDefiningOp<mlir::tensor::ExtractSliceOp>()) {
            EXPECT_EQ(slice.getStaticStrides(),
                      (llvm::ArrayRef<int64_t>{1, 1, 1}));
            ASSERT_GE(slice.getStaticOffsets()[1], 0);
            offset += slice.getStaticOffsets()[1];
            input = slice.getSource();
          } else if (auto cast = input.getDefiningOp<mlir::tensor::CastOp>()) {
            input = cast.getSource();
          } else if (auto argument =
                         mlir::dyn_cast<mlir::BlockArgument>(input)) {
            auto region = mlir::dyn_cast<wafer::TileRegionOp>(
                argument.getOwner()->getParentOp());
            if (!region)
              break;
            input = region.getInputs()[argument.getArgNumber()];
          } else {
            break;
          }
        }
        ASSERT_GE(offset, 0);
        ASSERT_LE(offset + inputType.getDimSize(1), extent);
        for (int64_t k = offset; k < offset + inputType.getDimSize(1); ++k)
          ++coverage[k];
      });
      EXPECT_EQ(contributions, static_cast<unsigned>(parts));
      EXPECT_TRUE(
          llvm::all_of(coverage, [](unsigned count) { return count == 1; }));
    }
  }
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
        score { ^bb0(%dot: f16, %scale_arg: f32):
          %wide = arith.extf %dot : f16 to f32
          %scaled = arith.mulf %wide, %scale_arg : f32
          wafer.linalg_ext.attention.yield %scaled : f32
        }
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
        algorithm(<flash_decoding>) indexing_maps = [#q, #k, #v, #s, #o] score {
    ^bb0(%attention_3_dot: f16, %attention_3_scale: f32):
      %attention_3_converted = arith.extf %attention_3_dot : f16 to f32
      %attention_3_scaled = arith.mulf %attention_3_converted, %attention_3_scale : f32
      wafer.linalg_ext.attention.yield %attention_3_scaled : f32
    }
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
        algorithm(<flash_decoding>) indexing_maps = [#q, #k, #v, #s, #o] score {
    ^bb0(%attention_4_dot: f16, %attention_4_scale: f32):
      %attention_4_converted = arith.extf %attention_4_dot : f16 to f32
      %attention_4_scaled = arith.mulf %attention_4_converted, %attention_4_scale : f32
      wafer.linalg_ext.attention.yield %attention_4_scaled : f32
    }
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

TEST(SpatialRegionMaterializationTest,
     RelationCoordinatedContributionsMaterialize) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  for (bool conv : {false, true})
    for (bool generic : {false, true})
      for (int64_t extent : {1024, 1025, 1031})
        for (unsigned mode : {0u, 1u, 2u, 3u, 4u}) {
          const bool reduction = mode != 0;
          const int64_t rows = mode == 4 ? 2 : mode == 3 ? 1 : extent;
          const int64_t depth = mode >= 3 ? extent : 128;
          const int64_t columns = mode >= 3 ? 16 : 128;
          SCOPED_TRACE(std::to_string(conv) + ":" + std::to_string(generic) +
                       ":" + std::to_string(extent) + ":" +
                       std::to_string(mode));
          auto context = createContext();
          auto source = mlir::parseSourceString<mlir::ModuleOp>(
              wafer::compiler::testing::spatialContractionSource(
                  rows, conv, generic, 16, depth, columns),
              context.get());
          ASSERT_TRUE(source);
          std::string detail;
          auto dag = analyzeSingleTensorProgram(*source, detail);
          ASSERT_TRUE(mlir::succeeded(dag)) << detail;
          auto topology = TargetTopology::create(*source, &detail);
          ASSERT_TRUE(mlir::succeeded(topology)) << detail;
          auto domain = buildSpatialPlanDomain(*dag, *topology, CardId(0));
          ASSERT_TRUE(domain.succeeded());
          auto seed = domain.domain->getFirstPlan();
          auto *consumerOp = dag->getFunction()
                                 .getBody()
                                 .front()
                                 .getTerminator()
                                 ->getOperand(0)
                                 .getDefiningOp();
          const auto &roots = domain.domain->getProblem().getSemanticRoots();
          auto producerKey =
              roots.find(consumerOp->getOperand(0).getDefiningOp())->key;
          auto consumerKey = roots.find(consumerOp)->key;
          auto producer = llvm::find_if(seed.nodes, [&](const auto &node) {
            return node.root == producerKey;
          });
          ASSERT_NE(producer, seed.nodes.end());
          producer->axes[reduction ? 2 : 1].parameter = mode == 4 ? 2 : 4;
          if (mode == 4)
            producer->axes[1].parameter = 2;
          producer->embedding = {TileId(0), TileId(1), TileId(2), TileId(3)};
          auto coordinated =
              propagateSpatialPartitions(*domain.domain, *dag, seed, {});
          ASSERT_TRUE(mlir::succeeded(coordinated));
          if (mode == 3) {
            // A distributed init is requested by every K contribution, but
            // each init fragment must be assembled only once at the merge.
            auto dps =
                mlir::cast<mlir::DestinationStyleOpInterface>(consumerOp);
            auto *initRoot =
                roots.find(dps.getDpsInitOperand(0)->get().getDefiningOp());
            ASSERT_NE(initRoot, nullptr);
            auto initNode = llvm::find_if(coordinated->nodes, [&](auto &node) {
              return node.root == initRoot->key;
            });
            ASSERT_NE(initNode, coordinated->nodes.end());
            initNode->axes[2].parameter = 16;
            initNode->embedding = allTiles();
          }
          auto consumer =
              llvm::find_if(coordinated->nodes, [&](const auto &node) {
                return node.root == consumerKey;
              });
          ASSERT_NE(consumer, coordinated->nodes.end());
          ASSERT_EQ(consumer->embedding.size(), 4u);
          if (mode >= 2) {
            ASSERT_EQ(consumer->reductionMerges.size(), mode == 4 ? 2u : 1u);
            for (auto &merge : consumer->reductionMerges)
              merge.tile = TileId(15);
          }
          auto actual = materializeWithSpatialPlan(
              *source, [&](SpatialPlan &plan) { plan = *coordinated; }, detail);
          ASSERT_TRUE(mlir::succeeded(actual)) << detail;
          EXPECT_TRUE(mlir::succeeded(mlir::verify(*actual->module)));
          unsigned repeatedFragmentCopies = 0;
          actual->module->walk([&](mlir::tensor::InsertSliceOp inserted) {
            auto previous =
                inserted.getDest().getDefiningOp<mlir::tensor::InsertSliceOp>();
            auto currentSlice =
                inserted.getSource()
                    .getDefiningOp<mlir::tensor::ExtractSliceOp>();
            auto previousSlice =
                previous ? previous.getSource()
                               .getDefiningOp<mlir::tensor::ExtractSliceOp>()
                         : mlir::tensor::ExtractSliceOp{};
            if (currentSlice && previousSlice &&
                currentSlice.getSource() == previousSlice.getSource() &&
                currentSlice.getMixedOffsets() ==
                    previousSlice.getMixedOffsets() &&
                currentSlice.getMixedSizes() == previousSlice.getMixedSizes() &&
                currentSlice.getMixedStrides() ==
                    previousSlice.getMixedStrides() &&
                inserted.getMixedOffsets() == previous.getMixedOffsets() &&
                inserted.getMixedSizes() == previous.getMixedSizes() &&
                inserted.getMixedStrides() == previous.getMixedStrides())
              ++repeatedFragmentCopies;
          });
          EXPECT_EQ(repeatedFragmentCopies, 0u);
          EXPECT_EQ(actual->relations.structuralOutputs.size(), mode == 4 ? 2u
                                                                : reduction
                                                                    ? 1u
                                                                    : 4u);
          unsigned multiplyBodies = 0;
          actual->module->walk([&](mlir::arith::MulFOp) { ++multiplyBodies; });
          EXPECT_EQ(multiplyBodies, 4u);
          if (reduction) {
            unsigned mergeBodies = 0;
            actual->module->walk([&](mlir::linalg::LinalgOp op) {
              if (op.getNumReductionLoops())
                ++mergeBodies;
            });
            EXPECT_EQ(mergeBodies, 4u);
          }
          if (mode >= 2) {
            EXPECT_GE(actual->relations.boundaryRelations.size(), 5u);
          }
          llvm::SmallVector<TileRegionOp, 16> regions;
          actual->module->walk(
              [&](TileRegionOp region) { regions.push_back(region); });
          unsigned loops = 0;
          for (auto region : regions) {
            auto temporal = buildTemporalDomain(region);
            ASSERT_TRUE(temporal.succeeded());
            auto first = temporal.domain->getFirstChoice();
            ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
            TemporalChoice choice = *first.getChoice();
            for (auto [scope, descriptor] : llvm::zip_equal(
                     choice.scopes, temporal.domain->getScopeDescriptors())) {
              for (size_t axis = 0; axis < scope.iteratorTileSizes.size();
                   ++axis)
                if (descriptor.iteratorCapabilities[axis] ==
                    IteratorTilingCapability::Tileable)
                  scope.iteratorTileSizes[axis] =
                      std::min<int64_t>(16, descriptor.iterationExtents[axis]);
              auto order = buildFirstTemporalLoopOrder(
                  descriptor.iterationExtents, scope.iteratorTileSizes,
                  descriptor.precedence);
              ASSERT_TRUE(mlir::succeeded(order));
              scope.loopOrder = std::move(*order);
            }
            ASSERT_TRUE(temporal.domain->contains(choice));
            TemporalTilingFailure failure;
            auto tiled = applyTemporalTiling({{*temporal.domain, choice}},
                                             actual->relations, &failure);
            ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
            loops += tiled->loops;
          }
          EXPECT_GT(loops, 0u);
          EXPECT_TRUE(mlir::succeeded(mlir::verify(*actual->module)));
          if (mode >= 2) {
            auto layout = resolveCurrentLayoutsAndBufferize(*actual->module,
                                                            actual->relations);
            ASSERT_TRUE(layout.succeeded()) << layout.detail;
            auto compute = lowerStructuredComputeToTile(*actual->module,
                                                        actual->relations);
            ASSERT_TRUE(compute.succeeded()) << compute.detail;
            auto movement = materializeTileBoundaryMovement(*actual->module,
                                                            actual->relations);
            ASSERT_TRUE(movement.succeeded()) << movement.detail;
            if (mode == 4) {
              for (auto tile : actual->module->getOps<TileModuleOp>())
                for (auto entry : tile.getOps<mlir::func::FuncOp>()) {
                  ASSERT_EQ(entry.getNumResults(), 1u);
                  auto returned = mlir::cast<mlir::func::ReturnOp>(
                      entry.getBody().front().getTerminator());
                  auto root = returned.getOperand(0)
                                  .getDefiningOp<mlir::memref::AllocOp>();
                  ASSERT_TRUE(root);
                  EXPECT_EQ(root.getType().getShape(),
                            (llvm::ArrayRef<int64_t>{1, rows, columns}));
                }
              continue;
            }
            wafer::frontend::FrontendProgramVerificationResult program;
            program.numPartitions = 1;
            program.programUserInputCount = 2;
            program.distributedInputs = {
                wafer::compiler::testing::boundary(0, {1, rows, depth}),
                wafer::compiler::testing::boundary(1, {1, depth, columns})};
            program.distributedOutputs = {
                wafer::compiler::testing::boundary(0, {1, rows, columns})};
            wafer::compiler::ProgramDataHandoff data;
            std::string diagnosticsText;
            llvm::raw_string_ostream diagnostics(diagnosticsText);
            CurrentIRDownstreamStatistics downstream;
            ExecutableLoweringStatistics executable;
            auto compiled = compileCurrentIRCandidateToExecutable(
                std::move(actual->module), std::move(actual->relations),
                CardId(0), allTiles(), program,
                wafer::compiler::testing::executionConfig(), diagnostics, data,
                {}, &downstream, &executable);
            EXPECT_EQ(executable.actualMemoryTargetGateInvocations, 1u);
            EXPECT_GT(downstream.instructionOperations, 0u);
            // Compact local partials now reach the actual memory/target gate
            // without the former expanded reduction tensor.
            ASSERT_TRUE(compiled.isAccepted())
                << compiled.gate << ": " << compiled.detail << "\n"
                << diagnosticsText;
            EXPECT_EQ(executable.deviceExecutablesProduced, 1u);
          }
        }
}

TEST(SpatialRegionMaterializationTest,
     ConvolutionPartialsKeepLocalWindowReductions) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  for (int64_t width : {1024, 1025}) {
    for (unsigned splitAxis : {4u, 5u, 6u}) {
      SCOPED_TRACE(std::to_string(width) + ":" + std::to_string(splitAxis));
      const unsigned parts = splitAxis == 6 ? 4 : 3;
      auto context = createContext();
      std::string input = "tensor<1x5x" + std::to_string(width + 2) + "x4xf16>";
      std::string output = "tensor<1x3x" + std::to_string(width) + "x8xf16>";
      std::string text;
      llvm::raw_string_ostream os(text);
      os << R"mlir(module {
  wafer.target.topology @target {card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>,
      unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%image: )mlir"
         << input << ", %weight: tensor<3x3x4x8xf16>, %init: " << output
         << ") -> " << output << " {\n"
         << "%result = linalg.conv_2d_nhwc_hwcf ins(%image, %weight : " << input
         << ", tensor<3x3x4x8xf16>) outs(%init : " << output << ") -> "
         << output << "\nreturn %result : " << output << "\n}}\n";
      auto source =
          mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
      ASSERT_TRUE(source);
      SpatialRegionMaterializationFailure failure;
      std::string detail;
      auto actual = materializeWithSpatialDomainPlan(
          *source,
          [&](const SpatialRootDomainFacts &root, SpatialPlan &plan,
              std::string &reason) {
            auto &node = plan.nodes.front();
            for (auto &axis : node.axes) {
              axis.scheme = IteratorPartitionScheme::BalancedParts;
              axis.parameter = axis.iterator == splitAxis ? parts : 1;
            }
            node.embedding.clear();
            for (unsigned tile = 0; tile != parts; ++tile)
              node.embedding.push_back(TileId(tile));
            auto groups =
                deriveSpatialReductionGroups(root, node.axes, &reason);
            if (mlir::failed(groups))
              return;
            node.reductionMerges.clear();
            for (const auto &group : *groups)
              node.reductionMerges.push_back({group, TileId(15)});
          },
          failure, detail);
      ASSERT_TRUE(mlir::succeeded(actual)) << detail;
      unsigned partials = 0;
      actual->module->walk([&](mlir::linalg::LinalgOp op) {
        if (!op.getNumReductionLoops())
          return;
        ++partials;
        EXPECT_EQ(op.getNumReductionLoops(), 3u);
        EXPECT_EQ(mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
                      .getShape(),
                  (llvm::ArrayRef<int64_t>{1, 3, width, 8}));
      });
      EXPECT_EQ(partials, parts);
      actual->module->walk([&](mlir::Operation *op) {
        for (mlir::Type type : op->getResultTypes()) {
          if (auto tensor = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
            EXPECT_LE(tensor.getRank(), 4);
          }
        }
      });
      auto layout =
          resolveCurrentLayoutsAndBufferize(*actual->module, actual->relations);
      ASSERT_TRUE(layout.succeeded()) << layout.detail;
      auto lowered =
          lowerStructuredComputeToTile(*actual->module, actual->relations);
      ASSERT_TRUE(lowered.succeeded()) << lowered.detail;
      EXPECT_EQ(countOps<ComputeConvOp>(*actual->module), parts);
      EXPECT_EQ(countOps<mlir::linalg::LinalgOp>(*actual->module), 0u);
      auto movement =
          materializeTileBoundaryMovement(*actual->module, actual->relations);
      ASSERT_TRUE(movement.succeeded()) << movement.detail;
      EXPECT_TRUE(mlir::succeeded(verifyPhysicalTileDataflow(*actual->module)));
    }
  }
}

TEST(SpatialRegionMaterializationTest, AssemblesHaloInExactConsumerWindow) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  for (int64_t extent : {1024, 1025, 1031})
    for (int64_t parts : {4, 16})
      for (bool generic : {false, true}) {
        SCOPED_TRACE(std::to_string(extent) + ":" + std::to_string(parts) +
                     ":" + std::to_string(generic));
        auto context = createContext();
        std::string text;
        llvm::raw_string_ostream out(text);
        const std::string inputType =
            "tensor<1x" + std::to_string(extent + 2) + "x8xf16>";
        const std::string outputType =
            "tensor<1x" + std::to_string(extent) + "x16xf16>";
        out << R"mlir(module {
  wafer.target.topology @target {card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>,
      unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: )mlir"
            << inputType
            << ", %weight: tensor<3x8x16xf16>, %init: " << outputType << ") -> "
            << outputType << " {\n"
            << "%empty = tensor.empty() : " << inputType << "\n"
            << "%producer = linalg.map ins(%input : " << inputType
            << ") outs(%empty : " << inputType << R"mlir() (%x: f16) {
    %zero = arith.constant 0.0 : f16
    %relu = arith.maximumf %x, %zero : f16
    linalg.yield %relu : f16
  }
  %result = )mlir";
        if (generic)
          out << R"mlir(linalg.generic {
    indexing_maps = [affine_map<(b, x, c, k, ic) -> (b, x + k, ic)>,
                     affine_map<(b, x, c, k, ic) -> (k, ic, c)>,
                     affine_map<(b, x, c, k, ic) -> (b, x, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
    )mlir";
        else
          out << "linalg.conv_1d_nwc_wcf {strides = dense<1> : tensor<1xi64>, "
                 "dilations = dense<1> : tensor<1xi64>} ";
        out << "ins(%producer, %weight : " << inputType
            << ", tensor<3x8x16xf16>) outs(%init : " << outputType << ")";
        if (generic)
          out << R"mlir( {
    ^bb0(%x: f16, %w: f16, %old: f16):
      %mul = arith.mulf %x, %w : f16
      %sum = arith.addf %mul, %old : f16
      linalg.yield %sum : f16
    })mlir";
        out << " -> " << outputType << "\nreturn %result : " << outputType
            << "\n}}\n";
        auto source =
            mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
        ASSERT_TRUE(source);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
        std::string detail;
        auto actual = materializeWithSpatialPlan(
            *source,
            [&](SpatialPlan &plan) {
              for (auto &node : plan.nodes) {
                for (auto &axis : node.axes) {
                  axis.scheme = IteratorPartitionScheme::BalancedParts;
                  axis.parameter = 1;
                }
                node.axes[1].parameter = parts;
                node.embedding.clear();
                for (int64_t tile = 0; tile < parts; ++tile)
                  node.embedding.push_back(TileId(tile));
                node.reductionMerges.clear();
              }
            },
            detail);
        ASSERT_TRUE(mlir::succeeded(actual)) << detail;
        unsigned consumers = 0;
        actual->module->walk([&](mlir::linalg::LinalgOp op) {
          if (op.getNumReductionLoops() == 0)
            return;
          ++consumers;
          int64_t tile = op->getParentOfType<TileModuleOp>().getTileId();
          const int64_t begin =
              tile * (extent / parts) + std::min(tile, extent % parts);
          const int64_t rows = extent / parts + (tile < extent % parts);
          auto value = op.getDpsInputOperand(0)->get();
          EXPECT_EQ(
              mlir::cast<mlir::RankedTensorType>(value.getType()).getShape(),
              (llvm::ArrayRef<int64_t>{1, rows + 2, 8}));
          // Independent interval oracle: every requested input row appears
          // exactly once, including halo from another spatial owner.
          std::vector<unsigned> coverage(rows + 2);
          unsigned fragments = 0;
          while (auto inserted =
                     value.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
            auto slice = inserted.getSource()
                             .getDefiningOp<mlir::tensor::ExtractSliceOp>();
            ASSERT_TRUE(slice);
            int64_t offset = inserted.getStaticOffsets()[1];
            int64_t count = inserted.getStaticSizes()[1];
            ASSERT_GE(offset, 0);
            ASSERT_LE(offset + count, rows + 2);
            EXPECT_EQ(slice.getStaticOffsets()[1], begin + offset);
            EXPECT_EQ(slice.getStaticSizes(), inserted.getStaticSizes());
            EXPECT_EQ(inserted.getStaticStrides(),
                      (llvm::ArrayRef<int64_t>{1, 1, 1}));
            for (int64_t row = offset; row < offset + count; ++row)
              ++coverage[row];
            ++fragments;
            value = inserted.getDest();
          }
          EXPECT_GE(fragments, 2u);
          EXPECT_TRUE(value.getDefiningOp<mlir::tensor::EmptyOp>());
          EXPECT_TRUE(llvm::all_of(coverage,
                                   [](unsigned count) { return count == 1; }));
          op->getParentOfType<TileRegionOp>().walk(
              [&](mlir::tensor::EmptyOp empty) {
                EXPECT_NE(empty.getType().getShape(),
                          (llvm::ArrayRef<int64_t>{1, extent + 2, 8}));
              });
        });
        EXPECT_EQ(consumers, parts);
        EXPECT_FALSE(actual->relations.boundaryRelations.empty());

        llvm::SmallVector<TileRegionOp, 16> regions;
        actual->module->walk(
            [&](TileRegionOp region) { regions.push_back(region); });
        unsigned loops = 0;
        for (auto region : regions) {
          auto temporal = buildTemporalDomain(region);
          ASSERT_TRUE(temporal.succeeded());
          auto first = temporal.domain->getFirstChoice();
          ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
          TemporalChoice choice = *first.getChoice();
          for (auto [scope, descriptor] : llvm::zip_equal(
                   choice.scopes, temporal.domain->getScopeDescriptors())) {
            for (size_t axis = 0; axis < scope.iteratorTileSizes.size(); ++axis)
              if (descriptor.iteratorCapabilities[axis] ==
                  IteratorTilingCapability::Tileable)
                scope.iteratorTileSizes[axis] =
                    std::min<int64_t>(16, descriptor.iterationExtents[axis]);
            auto order = buildFirstTemporalLoopOrder(
                descriptor.iterationExtents, scope.iteratorTileSizes,
                descriptor.precedence);
            ASSERT_TRUE(mlir::succeeded(order));
            scope.loopOrder = std::move(*order);
          }
          ASSERT_TRUE(temporal.domain->contains(choice));
          TemporalTilingFailure failure;
          auto tiled = applyTemporalTiling({{*temporal.domain, choice}},
                                           actual->relations, &failure);
          ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
          loops += tiled->loops;
        }
        EXPECT_GT(loops, parts);
        auto layout = resolveCurrentLayoutsAndBufferize(*actual->module,
                                                        actual->relations);
        ASSERT_TRUE(layout.succeeded()) << layout.detail;
        auto compute =
            lowerStructuredComputeToTile(*actual->module, actual->relations);
        ASSERT_TRUE(compute.succeeded()) << compute.detail;
        auto movement =
            materializeTileBoundaryMovement(*actual->module, actual->relations);
        ASSERT_TRUE(movement.succeeded()) << movement.detail;
        wafer::frontend::FrontendProgramVerificationResult program;
        program.numPartitions = 1;
        program.programUserInputCount = 3;
        program.distributedInputs = {
            wafer::compiler::testing::boundary(0, {1, extent + 2, 8}),
            wafer::compiler::testing::boundary(1, {3, 8, 16}),
            wafer::compiler::testing::boundary(2, {1, extent, 16})};
        program.distributedOutputs = {
            wafer::compiler::testing::boundary(0, {1, extent, 16})};
        wafer::compiler::ProgramDataHandoff data;
        std::string diagnosticsText;
        llvm::raw_string_ostream diagnostics(diagnosticsText);
        CurrentIRDownstreamStatistics downstream;
        ExecutableLoweringStatistics executable;
        auto compiled = compileCurrentIRCandidateToExecutable(
            std::move(actual->module), std::move(actual->relations), CardId(0),
            allTiles(), program, wafer::compiler::testing::executionConfig(),
            diagnostics, data, {}, &downstream, &executable);
        ASSERT_TRUE(compiled.isAccepted())
            << compiled.gate << ": " << compiled.detail << "\n"
            << diagnosticsText;
        EXPECT_EQ(executable.actualMemoryTargetGateInvocations, 1u);
        EXPECT_EQ(executable.deviceExecutablesProduced, 1u);
        EXPECT_GT(downstream.instructionOperations, 0u);
      }
}

TEST(SpatialRegionMaterializationTest, MaterializesSplatAtSelectedDemand) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  // F32 reproduces wide opmath constants exceeding SPM before slicing; ordinary
  // FP16/BF16 inputs exercise the same rule without changing arithmetic
  // formats.
  for (int64_t extent : {1024, 1025, 1031})
    for (llvm::StringRef dtype : {"f16", "bf16", "f32"})
      for (unsigned viewKind : {0u, 1u, 2u})
        for (int64_t parts : {4, 16}) {
          SCOPED_TRACE(std::to_string(extent) + ":" + dtype.str() + ":" +
                       std::to_string(viewKind) + ":" + std::to_string(parts));
          auto context = createContext();
          const std::string type = "tensor<1x" + std::to_string(extent) +
                                   "x3072x" + dtype.str() + ">";
          std::string text;
          llvm::raw_string_ostream out(text);
          out << R"mlir(module {
  wafer.target.topology @target {card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>,
      unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: )mlir"
              << type << ") -> " << type << " {\n";
          if (viewKind == 1) {
            const std::string expanded = "tensor<1x" + std::to_string(extent) +
                                         "x3x1024x" + dtype.str() + ">";
            out << "%literal = arith.constant dense<0.5> : " << expanded
                << "\n%c = tensor.collapse_shape %literal [[0], [1], [2, 3]] : "
                << expanded << " into " << type << "\n";
          } else if (viewKind == 2) {
            const std::string expanded = "tensor<1x1x" +
                                         std::to_string(extent) + "x3072x" +
                                         dtype.str() + ">";
            out << "%literal = arith.constant dense<-0.0> : " << type
                << "\n%expanded = tensor.expand_shape %literal [[0, 1], [2], "
                   "[3]] output_shape [1, 1, "
                << extent << ", 3072] : " << type << " into " << expanded
                << "\n%c = tensor.collapse_shape %expanded [[0, 1], [2], [3]] "
                   ": "
                << expanded << " into " << type << "\n";
          } else {
            out << "%c = arith.constant dense<0.5> : " << type << "\n";
          }
          out << "%empty = tensor.empty() : " << type << R"mlir(
  %result = linalg.generic {
    indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                     affine_map<(b, m, n) -> (b, m, n)>,
                     affine_map<(b, m, n) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%input, %c : )mlir"
              << type << ", " << type << ") outs(%empty : " << type
              << ") {\n^bb0(%x: " << dtype << ", %factor: " << dtype
              << ", %old: " << dtype
              << "):\n%y = arith.mulf %x, %factor : " << dtype
              << "\nlinalg.yield %y : " << dtype << "\n} -> " << type
              << "\nreturn %result : " << type << "\n}}\n";
          auto source =
              mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
          ASSERT_TRUE(source);
          ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
          std::string detail;
          auto actual = materializeWithSpatialPlan(
              *source,
              [&](SpatialPlan &plan) {
                for (auto &node : plan.nodes) {
                  for (auto &axis : node.axes) {
                    axis.scheme = IteratorPartitionScheme::BalancedParts;
                    axis.parameter = 1;
                  }
                  node.axes[1].parameter = 2;
                  node.axes[2].parameter = parts / 2;
                  node.embedding.clear();
                  for (int64_t tile = 0; tile < parts; ++tile)
                    node.embedding.push_back(TileId(tile));
                }
              },
              detail);
          ASSERT_TRUE(mlir::succeeded(actual)) << detail;
          unsigned fills = 0;
          int64_t elements = 0;
          std::set<std::pair<int64_t, int64_t>> windows;
          actual->module->walk([&](mlir::linalg::GenericOp op) {
            auto fill = op.getInputs()[1].getDefiningOp<mlir::linalg::FillOp>();
            ASSERT_TRUE(fill);
            auto local =
                mlir::cast<mlir::RankedTensorType>(fill.getResult(0).getType());
            EXPECT_EQ(local, op.getInputs()[0].getType());
            EXPECT_EQ(local.getDimSize(0), 1);
            EXPECT_TRUE(local.getDimSize(1) == extent / 2 ||
                        local.getDimSize(1) == (extent + 1) / 2);
            EXPECT_EQ(local.getDimSize(2), 3072 / (parts / 2));
            auto scalar =
                fill.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
            ASSERT_TRUE(scalar);
            EXPECT_EQ(scalar.getValue(),
                      mlir::FloatAttr::get(local.getElementType(),
                                           viewKind == 2 ? -0.0 : 0.5));
            auto region = op->getParentOfType<TileRegionOp>();
            auto input =
                op.getInputs()[0].getDefiningOp<mlir::tensor::ExtractSliceOp>();
            if (!input)
              input = region->getOperand(0)
                          .getDefiningOp<mlir::tensor::ExtractSliceOp>();
            ASSERT_TRUE(input);
            EXPECT_EQ(input.getStaticSizes(), local.getShape());
            EXPECT_EQ(input.getStaticStrides(),
                      (llvm::ArrayRef<int64_t>{1, 1, 1}));
            EXPECT_EQ(input.getStaticOffsets()[0], 0);
            EXPECT_EQ(local.getDimSize(1), input.getStaticOffsets()[1] == 0
                                               ? (extent + 1) / 2
                                               : extent / 2);
            EXPECT_TRUE(windows
                            .emplace(input.getStaticOffsets()[1],
                                     input.getStaticOffsets()[2])
                            .second);
            elements += local.getNumElements();
            ++fills;
          });
          EXPECT_EQ(fills, parts);
          for (int64_t row : {int64_t{0}, (extent + 1) / 2})
            for (int64_t column = 0; column < 3072;
                 column += 3072 / (parts / 2))
              EXPECT_EQ(windows.count({row, column}), 1u);
          EXPECT_EQ(elements, extent * 3072);
          EXPECT_EQ(countOps<mlir::linalg::FillOp>(*actual->module), parts);

          // At 16 Tiles the selected spatial window already fits. At 4 Tiles
          // it requires multiple temporal blocks and non-divisible row tails.
          {
            llvm::SmallVector<TileRegionOp> regions;
            actual->module->walk(
                [&](TileRegionOp region) { regions.push_back(region); });
            unsigned loops = 0;
            unsigned tails = 0;
            for (auto region : regions) {
              auto temporal = buildTemporalDomain(region);
              ASSERT_TRUE(temporal.succeeded());
              auto first = temporal.domain->getFirstChoice();
              ASSERT_EQ(first.getKind(), TemporalSuccessorKind::Choice);
              TemporalChoice choice = *first.getChoice();
              for (auto [scope, descriptor] : llvm::zip_equal(
                       choice.scopes, temporal.domain->getScopeDescriptors())) {
                for (size_t axis = 0; axis < scope.iteratorTileSizes.size();
                     ++axis)
                  if (parts == 4 && descriptor.iteratorCapabilities[axis] ==
                                        IteratorTilingCapability::Tileable)
                    scope.iteratorTileSizes[axis] = std::min<int64_t>(
                        32, descriptor.iterationExtents[axis]);
                auto order = buildFirstTemporalLoopOrder(
                    descriptor.iterationExtents, scope.iteratorTileSizes,
                    descriptor.precedence);
                ASSERT_TRUE(mlir::succeeded(order));
                scope.loopOrder = std::move(*order);
              }
              ASSERT_TRUE(temporal.domain->contains(choice));
              TemporalTilingFailure failure;
              auto tiled = applyTemporalTiling({{*temporal.domain, choice}},
                                               actual->relations, &failure);
              ASSERT_TRUE(mlir::succeeded(tiled)) << failure.detail;
              loops += tiled->loops;
              tails += tiled->specializedTails;
            }
            if (parts == 4) {
              EXPECT_GT(loops, parts);
              if (extent % 2 != 0) {
                EXPECT_GT(tails, 0u);
              }
            } else {
              EXPECT_EQ(loops, 0u);
            }
          }
          auto layout = resolveCurrentLayoutsAndBufferize(*actual->module,
                                                          actual->relations);
          ASSERT_TRUE(layout.succeeded()) << layout.detail;
          auto compute =
              lowerStructuredComputeToTile(*actual->module, actual->relations);
          ASSERT_TRUE(compute.succeeded()) << compute.detail;
          auto movement = materializeTileBoundaryMovement(*actual->module,
                                                          actual->relations);
          ASSERT_TRUE(movement.succeeded()) << movement.detail;
          wafer::frontend::FrontendProgramVerificationResult program;
          program.numPartitions = 1;
          program.programUserInputCount = 1;
          program.distributedInputs = {
              wafer::compiler::testing::boundary(0, {1, extent, 3072})};
          program.distributedOutputs = {
              wafer::compiler::testing::boundary(0, {1, extent, 3072})};
          wafer::compiler::ProgramDataHandoff data;
          std::string diagnosticsText;
          llvm::raw_string_ostream diagnostics(diagnosticsText);
          CurrentIRDownstreamStatistics downstream;
          ExecutableLoweringStatistics executable;
          auto compiled = compileCurrentIRCandidateToExecutable(
              std::move(actual->module), std::move(actual->relations),
              CardId(0), allTiles(), program,
              wafer::compiler::testing::executionConfig(), diagnostics, data,
              {}, &downstream, &executable);
          ASSERT_TRUE(compiled.isAccepted())
              << compiled.gate << ": " << compiled.detail << "\n"
              << diagnosticsText;
          EXPECT_EQ(executable.actualMemoryTargetGateInvocations, 1u);
          EXPECT_EQ(executable.deviceExecutablesProduced, 1u);
          EXPECT_GT(downstream.instructionOperations, 0u);
        }
}

TEST(SpatialRegionMaterializationTest, PreservesSharedAndFullConstantUses) {
  using namespace wafer;
  using namespace wafer::compiler::detail;
  for (bool splat : {false, true})
    for (bool partition : {false, true}) {
      SCOPED_TRACE(std::to_string(splat) + ":" + std::to_string(partition));
      auto context = createContext();
      auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.target.topology @target {card_grid = array<i64: 1, 1>,
      card_interconnect = "mesh", tile_grid = array<i64: 4, 4>,
      unavailable_tiles = array<i64>}
  wafer.execution.mesh @logical {axes = ["card"], shape = array<i64: 1>}
  func.func @main(%input: tensor<1x1025x128xf16>)
      -> (tensor<1x1025x128xf16>, tensor<1x1025x128xf16>) {
    %c = arith.constant dense<2.0> : tensor<1x1025x128xf16>
    %empty = tensor.empty() : tensor<1x1025x128xf16>
    %mul = linalg.mul ins(%input, %c : tensor<1x1025x128xf16>, tensor<1x1025x128xf16>)
        outs(%empty : tensor<1x1025x128xf16>) -> tensor<1x1025x128xf16>
    %pow = linalg.generic {
        indexing_maps = [affine_map<(b, m, n) -> (b, m, n)>,
                         affine_map<(b, m, n) -> (b, m, n)>,
                         affine_map<(b, m, n) -> (b, m, n)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input, %c : tensor<1x1025x128xf16>, tensor<1x1025x128xf16>)
        outs(%empty : tensor<1x1025x128xf16>) {
      ^bb0(%x: f16, %factor: f16, %old: f16):
        %p = math.powf %x, %factor : f16
        linalg.yield %p : f16
    } -> tensor<1x1025x128xf16>
    return %mul, %pow : tensor<1x1025x128xf16>, tensor<1x1025x128xf16>
  }
})mlir",
                                                            context.get());
      ASSERT_TRUE(source);
      auto literal = *source->getOps<mlir::func::FuncOp>().begin();
      auto constant = *literal.getOps<mlir::arith::ConstantOp>().begin();
      auto type = mlir::cast<mlir::RankedTensorType>(constant.getType());
      if (!splat) {
        llvm::SmallVector<llvm::APFloat> values;
        for (int64_t i = 0; i < type.getNumElements(); ++i)
          values.push_back(
              mlir::FloatAttr::get(type.getElementType(), i % 2 ? 3.0 : 2.0)
                  .getValue());
        constant.setValueAttr(mlir::DenseElementsAttr::get(type, values));
      }
      auto original = constant.getValue();
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*source)));
      std::string detail;
      auto actual = materializeWithSpatialPlan(
          *source,
          [&](SpatialPlan &plan) {
            for (auto [index, node] : llvm::enumerate(plan.nodes)) {
              for (auto &axis : node.axes) {
                axis.scheme = IteratorPartitionScheme::BalancedParts;
                axis.parameter = 1;
              }
              if (partition)
                node.axes[index == 0 ? 1 : 2].parameter = 4;
              node.embedding.clear();
              for (int64_t tile = 0; tile < (partition ? 4 : 1); ++tile)
                node.embedding.push_back(TileId(tile));
            }
          },
          detail);
      ASSERT_TRUE(mlir::succeeded(actual)) << detail;
      unsigned consumers = 0;
      actual->module->walk([&](mlir::linalg::LinalgOp op) {
        if (mlir::isa<mlir::linalg::FillOp>(op))
          return;
        auto value = op.getDpsInputOperand(1)->get();
        if (splat) {
          auto fill = value.getDefiningOp<mlir::linalg::FillOp>();
          ASSERT_TRUE(fill);
          EXPECT_EQ(fill.getResult(0).getType(),
                    op.getDpsInputOperand(0)->get().getType());
          auto scalar =
              fill.getInputs()[0].getDefiningOp<mlir::arith::ConstantOp>();
          ASSERT_TRUE(scalar);
          EXPECT_EQ(scalar.getValue(),
                    mlir::FloatAttr::get(type.getElementType(), 2.0));
          if (!partition) {
            EXPECT_EQ(fill.getResult(0).getType(), type);
          }
        } else {
          if (auto slice = value.getDefiningOp<mlir::tensor::ExtractSliceOp>())
            value = slice.getSource();
          auto preserved = value.getDefiningOp<mlir::arith::ConstantOp>();
          ASSERT_TRUE(preserved);
          EXPECT_EQ(preserved.getValue(), original);
        }
        op->walk([&](mlir::math::PowFOp power) {
          auto scalar = power.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
          if (splat) {
            ASSERT_TRUE(scalar);
            EXPECT_EQ(scalar.getValue(),
                      mlir::FloatAttr::get(type.getElementType(), 2.0));
          } else {
            EXPECT_EQ(power.getRhs(), op.getRegionInputArgs()[1]);
          }
        });
        ++consumers;
      });
      EXPECT_EQ(consumers, partition ? 8u : 2u);
      EXPECT_EQ(countOps<mlir::linalg::FillOp>(*actual->module),
                splat ? consumers : 0u);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*actual->module)));
      EXPECT_TRUE(mlir::succeeded(checkStructuredBufferRelationsCurrent(
          actual->module->getOperation(), actual->relations)));
    }
}

TEST(SpatialRegionMaterializationTest, SameTileOutputPiecesMustBeDisjoint) {
  auto context = createContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  wafer.tile.module card_id = 0 tile_id = 0 {
    func.func @entry(%input: tensor<1x1024x128xf16>) {
      %a = wafer.tile.region(%input : tensor<1x1024x128xf16>) -> (tensor<1x1025x128xf16>) {
      ^bb0(%x: tensor<1x1024x128xf16>):
        %e = tensor.empty() : tensor<1x1025x128xf16>
        %r = tensor.insert_slice %x into %e[0,0,0] [1,1024,128] [1,1,1] : tensor<1x1024x128xf16> into tensor<1x1025x128xf16>
        wafer.tile.yield %r : tensor<1x1025x128xf16>
      }
      %b = wafer.tile.region(%input : tensor<1x1024x128xf16>) -> (tensor<1x1025x128xf16>) {
      ^bb0(%x: tensor<1x1024x128xf16>):
        %e = tensor.empty() : tensor<1x1025x128xf16>
        %r = tensor.insert_slice %x into %e[0,0,0] [1,1024,128] [1,1,1] : tensor<1x1024x128xf16> into tensor<1x1025x128xf16>
        wafer.tile.yield %r : tensor<1x1025x128xf16>
      }
      return
    }
  }
}
)mlir",
                                                        context.get());
  ASSERT_TRUE(module);
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
  wafer::StructuredMaterializationRelations relations;
  module->walk([&](wafer::TileRegionOp region) {
    relations.structuralOutputs.push_back({0, region.getResult(0)});
  });
  ASSERT_TRUE(mlir::succeeded(
      wafer::compiler::detail::checkStructuredBufferRelationsCurrent(
          *module, relations)));
  auto layout = wafer::compiler::detail::resolveCurrentLayoutsAndBufferize(
      *module, relations);
  EXPECT_EQ(layout.status,
            wafer::compiler::detail::ExactPBQPStatus::NoSolution);
  EXPECT_EQ(layout.detail, "same-Tile output pieces overlap");
  EXPECT_EQ(layout.statistics.bufferizationInvocations, 0u);
}

} // namespace
