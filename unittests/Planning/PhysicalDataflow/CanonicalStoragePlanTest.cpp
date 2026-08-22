//===- CanonicalStoragePlanTest.cpp ----------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"

#include "TestSupport/Planning/CanonicalPlanningTestSupport.h"
#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <utility>

namespace {

using wafer::TileId;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

struct CanonicalStorageInputs {
  wafer::test::CanonicalPlanningPrefix prefix;
  CanonicalRepresentationCoordinate representations;
  CanonicalMovementCoordinate movements;
  SerializedExecutionPlan serialized;
};

class CanonicalStoragePlanTest : public ::testing::Test {
protected:
  CanonicalStoragePlanTest() {
    wafer::registerWaferCoreDialects(registry);
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

  static mlir::func::FuncOp function(mlir::ModuleOp module) {
    return *module.getOps<mlir::func::FuncOp>().begin();
  }

  static llvm::SmallVector<TileId, 16> allTiles() {
    llvm::SmallVector<TileId, 16> tiles;
    for (int64_t tile = 0; tile < 16; ++tile)
      tiles.push_back(TileId(tile));
    return tiles;
  }

  static std::string print(mlir::Operation *operation) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
  }

  static ExactIndexSet box(llvm::ArrayRef<int64_t> offsets,
                           llvm::ArrayRef<int64_t> sizes) {
    IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
    EXPECT_TRUE(set.isExact());
    StaticRectangularIndexSet rectangle{llvm::to_vector(offsets),
                                        llvm::to_vector(sizes)};
    return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion,
                         {rectangle});
  }

  static std::string mapSource(int64_t extent) {
    std::string source;
    llvm::raw_string_ostream stream(source);
    stream << R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @map(%input: tensor<2x)mlir"
           << extent << "x128xf16>) -> tensor<2x" << extent << "x128xf16> {\n"
           << "    %empty = tensor.empty() : tensor<2x" << extent
           << "x128xf16>\n"
           << "    %result = linalg.generic {\n"
           << "        indexing_maps = [#id, #id],\n"
           << "        iterator_types = [\"parallel\", \"parallel\", "
              "\"parallel\"]}\n"
           << "        ins(%input : tensor<2x" << extent
           << "x128xf16>) outs(%empty : tensor<2x" << extent << "x128xf16>) {\n"
           << "      ^bb0(%value: f16, %old: f16):\n"
           << "        linalg.yield %value : f16\n"
           << "    } -> tensor<2x" << extent << "x128xf16>\n"
           << "    return %result : tensor<2x" << extent << "x128xf16>\n"
           << "  }\n"
           << "}\n";
    return source;
  }

  mlir::FailureOr<CanonicalStorageInputs>
  buildInputs(const StructuredDAGAnalysis &dag, llvm::ArrayRef<TileId> tiles,
              std::string *failureReason) {
    auto prefix =
        wafer::test::buildCanonicalPlanningPrefix(dag, tiles, failureReason);
    if (mlir::failed(prefix))
      return mlir::failure();
    CanonicalRepresentationPlanOutcome representationOutcome =
        buildCanonicalRepresentationPlan(prefix->regions, prefix->temporal,
                                         prefix->rootWorks);
    const CanonicalRepresentationCoordinate *representations =
        getCanonicalRepresentationCoordinate(representationOutcome);
    if (!representations)
      return mlir::failure();
    CanonicalMovementPlanOutcome movementOutcome = buildCanonicalMovementPlan(
        prefix->regions, *representations, prefix->rootWorks);
    const CanonicalMovementCoordinate *movements =
        getCanonicalMovementCoordinate(movementOutcome);
    if (!movements)
      return mlir::failure();
    CanonicalSerializedExecutionPlanOutcome serializedOutcome =
        buildCanonicalSerializedExecutionPlan(prefix->regions,
                                              prefix->temporal);
    const SerializedExecutionPlan *serialized =
        getSerializedExecutionPlan(serializedOutcome);
    if (!serialized)
      return mlir::failure();
    return CanonicalStorageInputs{std::move(*prefix), *representations,
                                  *movements, *serialized};
  }

  static const StorageLifetimeDescription *
  findLifetime(const CanonicalStorageCoordinate &coordinate,
               const StorageObjectId &object) {
    auto lifetime = llvm::find_if(coordinate.lifetimes,
                                  [&](const StorageLifetimeDescription &entry) {
                                    return entry.object == object;
                                  });
    return lifetime == coordinate.lifetimes.end() ? nullptr : &*lifetime;
  }

  mlir::DialectRegistry registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(CanonicalStoragePlanTest,
       AlignedAndRaggedLoadsAndPublicationsHaveExactSingleSlots) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    auto module = parse(mapSource(extent));
    ASSERT_TRUE(module);
    const std::string before = print(module->getOperation());
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    CanonicalStoragePlanOutcome outcome = buildCanonicalStoragePlan(
        inputs->representations, inputs->movements, inputs->serialized);
    const CanonicalStorageCoordinate *coordinate =
        getCanonicalStorageCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(coordinate->plan.versionBindings.size(), 32u);
    ASSERT_EQ(coordinate->plan.storageObjects.size(), 32u);
    EXPECT_TRUE(coordinate->plan.gatherStagingBindings.empty());
    EXPECT_EQ(coordinate->resources.size(), 32u);
    EXPECT_EQ(coordinate->lifetimes.size(), 32u);

    unsigned boundaryObjects = 0;
    unsigned resultObjects = 0;
    llvm::SmallVector<int64_t, 32> querySizes;
    for (const StorageLifetimeDescription &lifetime : coordinate->lifetimes) {
      const auto *version =
          std::get_if<PhysicalVersionId>(&lifetime.object.origin);
      ASSERT_NE(version, nullptr);
      ASSERT_EQ(lifetime.uses.size(), 1u);
      if (std::holds_alternative<BoundaryRegionValueId>(
              version->logicalValue)) {
        ++boundaryObjects;
        const auto *definition =
            std::get_if<MovementActionId>(&lifetime.definition);
        ASSERT_NE(definition, nullptr);
        EXPECT_TRUE(std::holds_alternative<ExternalLoadId>(*definition));
        EXPECT_TRUE(
            std::holds_alternative<ExecutionInstanceId>(lifetime.uses.front()));
      } else if (std::holds_alternative<ExecutionResultValueId>(
                     version->logicalValue)) {
        ++resultObjects;
        EXPECT_TRUE(
            std::holds_alternative<ExecutionInstanceId>(lifetime.definition));
        const auto *use = std::get_if<MovementActionId>(&lifetime.uses.front());
        ASSERT_NE(use, nullptr);
        EXPECT_TRUE(std::holds_alternative<ResultPublicationId>(*use));
      }
    }
    EXPECT_EQ(boundaryObjects, 16u);
    EXPECT_EQ(resultObjects, 16u);
    for (const StorageResourceDescription &resource : coordinate->resources) {
      EXPECT_TRUE(resource.elementType.isF16());
      EXPECT_EQ(resource.encoding, wafer::MemLayout::Tensor);
      ASSERT_EQ(resource.exactDomain.getBoxes().size(), 1u);
      querySizes.push_back(resource.exactDomain.getBoxes().front().sizes[1]);
    }
    if (extent == 1024)
      EXPECT_EQ(*llvm::min_element(querySizes), *llvm::max_element(querySizes));
    else
      EXPECT_LT(*llvm::min_element(querySizes), *llvm::max_element(querySizes));

    std::reverse(inputs->representations.plan.primaryVersions.begin(),
                 inputs->representations.plan.primaryVersions.end());
    std::reverse(inputs->representations.resources.begin(),
                 inputs->representations.resources.end());
    std::reverse(inputs->movements.plan.externalLoads.begin(),
                 inputs->movements.plan.externalLoads.end());
    std::reverse(inputs->movements.plan.publications.begin(),
                 inputs->movements.plan.publications.end());
    std::reverse(inputs->movements.resources.begin(),
                 inputs->movements.resources.end());
    std::reverse(inputs->serialized.executions.begin(),
                 inputs->serialized.executions.end());
    CanonicalStoragePlanOutcome reversed = buildCanonicalStoragePlan(
        inputs->representations, inputs->movements, inputs->serialized);
    const CanonicalStorageCoordinate *reversedCoordinate =
        getCanonicalStorageCoordinate(reversed);
    ASSERT_NE(reversedCoordinate, nullptr);
    ASSERT_EQ(reversedCoordinate->lifetimes.size(),
              coordinate->lifetimes.size());
    for (auto [expected, actual] : llvm::zip_equal(
             coordinate->lifetimes, reversedCoordinate->lifetimes)) {
      EXPECT_EQ(actual.object, expected.object);
      EXPECT_EQ(actual.definition, expected.definition);
      EXPECT_EQ(actual.uses, expected.uses);
    }
    EXPECT_EQ(print(module->getOperation()), before);
  }
}

TEST_F(CanonicalStoragePlanTest,
       DiamondKeepsSharedSourceLiveAcrossEveryDDRTransfer) {
  auto module = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @diamond(%input: tensor<2x1025x128xf16>)
      -> (tensor<2x1025x128xf16>, tensor<2x1025x128xf16>) {
    %a0 = tensor.empty() : tensor<2x1025x128xf16>
    %a = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%input : tensor<2x1025x128xf16>)
        outs(%a0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    %b0 = tensor.empty() : tensor<2x1025x128xf16>
    %b = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x1025x128xf16>)
        outs(%b0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    %c0 = tensor.empty() : tensor<2x1025x128xf16>
    %c = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a : tensor<2x1025x128xf16>)
        outs(%c0 : tensor<2x1025x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1025x128xf16>
    return %b, %c : tensor<2x1025x128xf16>, tensor<2x1025x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
  CanonicalStoragePlanOutcome outcome = buildCanonicalStoragePlan(
      inputs->representations, inputs->movements, inputs->serialized);
  const CanonicalStorageCoordinate *coordinate =
      getCanonicalStorageCoordinate(outcome);
  ASSERT_NE(coordinate, nullptr);
  EXPECT_EQ(coordinate->plan.storageObjects.size(),
            inputs->representations.plan.primaryVersions.size());
  EXPECT_TRUE(coordinate->plan.gatherStagingBindings.empty());

  unsigned sharedSources = 0;
  unsigned ddrDefinitions = 0;
  for (const StorageLifetimeDescription &lifetime : coordinate->lifetimes) {
    unsigned ddrUses =
        llvm::count_if(lifetime.uses, [](const StorageAccessSite &site) {
          const auto *action = std::get_if<MovementActionId>(&site);
          return action &&
                 std::holds_alternative<DDRBoundaryTransferId>(*action);
        });
    sharedSources += ddrUses > 1;
    const auto *definition =
        std::get_if<MovementActionId>(&lifetime.definition);
    ddrDefinitions +=
        definition &&
        std::holds_alternative<DDRBoundaryTransferId>(*definition);
  }
  EXPECT_GT(sharedSources, 0u);
  EXPECT_EQ(ddrDefinitions, inputs->movements.plan.ddrTransfers.size());
}

TEST_F(CanonicalStoragePlanTest,
       FlashDecodingUsesOneStagingObjectPerRemoteComponent) {
  for (const auto &[queryExtent, keyValueExtent] :
       {std::pair<int64_t, int64_t>{1024, 1024}, {1025, 1031}}) {
    SCOPED_TRACE(queryExtent);
    auto module = parse(wafer::test::buildFlashDecodingPlanningFixture(
        queryExtent, keyValueExtent));
    ASSERT_TRUE(module);
    std::string failureReason;
    auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
    ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
    auto inputs = buildInputs(*dag, allTiles(), &failureReason);
    ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;
    CanonicalStoragePlanOutcome outcome = buildCanonicalStoragePlan(
        inputs->representations, inputs->movements, inputs->serialized);
    const CanonicalStorageCoordinate *coordinate =
        getCanonicalStorageCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(coordinate->plan.gatherStagingBindings.size(),
              inputs->movements.plan.reductionGathers.size());
    EXPECT_EQ(coordinate->plan.storageObjects.size(),
              inputs->representations.plan.primaryVersions.size() +
                  inputs->movements.plan.reductionGathers.size());

    std::set<PhysicalVersionId> gatheredSources;
    for (const ReductionGatherPlan &gather :
         inputs->movements.plan.reductionGathers)
      gatheredSources.insert(gather.source);
    unsigned localComponents = 0;
    unsigned maximumF32Staging = 0;
    unsigned accumulatorF16Staging = 0;
    for (const ReductionGatherStorageBinding &binding :
         coordinate->plan.gatherStagingBindings) {
      const StorageLifetimeDescription *lifetime =
          findLifetime(*coordinate, binding.stagingObject);
      ASSERT_NE(lifetime, nullptr);
      const auto *definition =
          std::get_if<MovementActionId>(&lifetime->definition);
      ASSERT_NE(definition, nullptr);
      const auto *gather = std::get_if<ReductionGatherId>(definition);
      ASSERT_NE(gather, nullptr);
      EXPECT_EQ(*gather, binding.gather);
      ASSERT_EQ(lifetime->uses.size(), 1u);
      EXPECT_TRUE(
          std::holds_alternative<ExecutionInstanceId>(lifetime->uses.front()));
      auto resource = llvm::find_if(
          coordinate->resources, [&](const StorageResourceDescription &entry) {
            return entry.object == binding.stagingObject;
          });
      ASSERT_NE(resource, coordinate->resources.end());
      ASSERT_EQ(resource->exactDomain.getBoxes().size(), 1u);
      const auto *component =
          std::get_if<CoupledComponentValueId>(&binding.gather.value);
      ASSERT_NE(component, nullptr);
      maximumF32Staging += component->component ==
                               wafer::CoupledReductionComponentKind::Maximum &&
                           resource->elementType.isF32();
      accumulatorF16Staging +=
          component->component ==
              wafer::CoupledReductionComponentKind::Accumulator &&
          resource->elementType.isF16();
    }
    for (const PhysicalVersionPlan &version :
         inputs->representations.plan.primaryVersions) {
      const auto *component =
          std::get_if<CoupledComponentValueId>(&version.id.logicalValue);
      if (!component ||
          !std::holds_alternative<RequiredRootExecution>(
              component->execution.source) ||
          gatheredSources.count(version.id))
        continue;
      const StorageLifetimeDescription *lifetime = findLifetime(
          *coordinate, StorageObjectId{StorageObjectOrigin{version.id}});
      ASSERT_NE(lifetime, nullptr);
      EXPECT_TRUE(
          llvm::any_of(lifetime->uses, [](const StorageAccessSite &site) {
            return std::holds_alternative<ExecutionInstanceId>(site);
          }));
      ++localComponents;
    }
    EXPECT_GT(localComponents, 0u);
    EXPECT_GT(maximumF32Staging, 0u);
    EXPECT_GT(accumulatorF16Staging, 0u);
  }
}

TEST_F(CanonicalStoragePlanTest,
       OrdinaryReductionSeparatesRemoteStagingFromLocalPartial) {
  for (int64_t extent : {1024, 1025}) {
    SCOPED_TRACE(extent);
    SemanticRootKey root;
    RootRegionWorkId remoteWork{root, TileId(0)};
    RootRegionWorkId mergeWork{root, TileId(1)};
    LogicalShardId remoteShard{root, {0}};
    LogicalShardId localShard{root, {1}};
    ReductionGroupId group{root, 0, {0}};
    ExecutionInstanceId remoteExecution{
        RequiredRootExecution{remoteWork, remoteShard}};
    ExecutionInstanceId localExecution{
        RequiredRootExecution{mergeWork, localShard}};
    ExecutionInstanceId mergeExecution{
        RequiredMergeExecution{mergeWork, group}};
    PhysicalVersionId remoteVersion{
        ReductionPartialValueId{remoteExecution, group, 0}};
    PhysicalVersionId localVersion{
        ReductionPartialValueId{localExecution, group, 0}};
    PhysicalVersionId resultVersion{ExecutionResultValueId{mergeExecution, 0}};

    ExactIndexSet domain = box({0, 0, 0}, {2, extent, 128});
    mlir::Type f16 = mlir::Float16Type::get(context.get());
    CanonicalRepresentationCoordinate representations;
    for (const PhysicalVersionId &version :
         {remoteVersion, localVersion, resultVersion}) {
      representations.plan.primaryVersions.push_back(
          {version, wafer::MemLayout::Tensor});
      representations.resources.push_back(
          {version, domain, f16, wafer::MemLayout::Tensor});
    }

    ReductionGatherId gatherId{
        group, remoteShard, ReductionPartialValueId{remoteExecution, group, 0}};
    ResultPublicationId publicationId{
        ExecutionResultValueId{mergeExecution, 0}};
    CanonicalMovementCoordinate movements;
    movements.plan.reductionGathers.push_back(
        {gatherId, remoteVersion, mergeExecution});
    movements.plan.publications.push_back({publicationId, resultVersion});
    movements.resources.push_back(
        {gatherId, domain, f16, TileId(0), TileId(1)});
    movements.resources.push_back(
        {publicationId, domain, f16, TileId(1), std::nullopt});
    SerializedExecutionPlan serialized{
        {remoteExecution, localExecution, mergeExecution}};

    CanonicalStoragePlanOutcome outcome =
        buildCanonicalStoragePlan(representations, movements, serialized);
    const CanonicalStorageCoordinate *coordinate =
        getCanonicalStorageCoordinate(outcome);
    ASSERT_NE(coordinate, nullptr);
    ASSERT_EQ(coordinate->plan.versionBindings.size(), 3u);
    ASSERT_EQ(coordinate->plan.gatherStagingBindings.size(), 1u);
    ASSERT_EQ(coordinate->plan.storageObjects.size(), 4u);
    const StorageObjectId stagingObject =
        coordinate->plan.gatherStagingBindings.front().stagingObject;
    auto stagingResource = llvm::find_if(
        coordinate->resources, [&](const StorageResourceDescription &resource) {
          return resource.object == stagingObject;
        });
    ASSERT_NE(stagingResource, coordinate->resources.end());
    ASSERT_EQ(stagingResource->exactDomain.getBoxes().size(), 1u);
    EXPECT_EQ(stagingResource->exactDomain.getBoxes().front().sizes,
              (llvm::SmallVector<int64_t, 3>{2, extent, 128}));

    const StorageLifetimeDescription *remote = findLifetime(
        *coordinate, StorageObjectId{StorageObjectOrigin{remoteVersion}});
    const StorageLifetimeDescription *local = findLifetime(
        *coordinate, StorageObjectId{StorageObjectOrigin{localVersion}});
    ASSERT_NE(remote, nullptr);
    ASSERT_NE(local, nullptr);
    ASSERT_EQ(remote->uses.size(), 1u);
    const auto *remoteUse =
        std::get_if<MovementActionId>(&remote->uses.front());
    ASSERT_NE(remoteUse, nullptr);
    EXPECT_TRUE(std::holds_alternative<ReductionGatherId>(*remoteUse));
    ASSERT_EQ(local->uses.size(), 1u);
    EXPECT_EQ(std::get<ExecutionInstanceId>(local->uses.front()),
              mergeExecution);
  }
}

TEST_F(CanonicalStoragePlanTest,
       RankZeroAndSupportValuesKeepTheirTypedLifetimes) {
  auto scalar = parse(R"mlir(
#scalar = affine_map<() -> ()>
module {
  func.func @scalar(%input: tensor<f16>) -> tensor<f16> {
    %empty = tensor.empty() : tensor<f16>
    %result = linalg.generic {indexing_maps = [#scalar, #scalar],
        iterator_types = []} ins(%input : tensor<f16>)
        outs(%empty : tensor<f16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<f16>
    return %result : tensor<f16>
  }
}
)mlir");
  ASSERT_TRUE(scalar);
  std::string failureReason;
  auto scalarDag =
      StructuredDAGAnalysis::create(function(*scalar), &failureReason);
  ASSERT_TRUE(mlir::succeeded(scalarDag)) << failureReason;
  llvm::SmallVector<TileId, 1> tile{TileId(0)};
  auto scalarInputs = buildInputs(*scalarDag, tile, &failureReason);
  ASSERT_TRUE(mlir::succeeded(scalarInputs)) << failureReason;
  CanonicalStoragePlanOutcome scalarOutcome = buildCanonicalStoragePlan(
      scalarInputs->representations, scalarInputs->movements,
      scalarInputs->serialized);
  const CanonicalStorageCoordinate *scalarCoordinate =
      getCanonicalStorageCoordinate(scalarOutcome);
  ASSERT_NE(scalarCoordinate, nullptr);
  ASSERT_EQ(scalarCoordinate->resources.size(), 2u);
  for (const StorageResourceDescription &resource : scalarCoordinate->resources)
    EXPECT_EQ(resource.exactDomain.getRank(), 0u);

  auto support = parse(R"mlir(
#id = affine_map<(b, m, n) -> (b, m, n)>
module {
  func.func @pad(%input: tensor<2x1025x128xf16>)
      -> tensor<2x1026x128xf16> {
    %zero = arith.constant 0.0 : f16
    %padded = tensor.pad %input low[0, 1, 0] high[0, 0, 0] {
      ^bb0(%b: index, %m: index, %n: index):
        tensor.yield %zero : f16
    } : tensor<2x1025x128xf16> to tensor<2x1026x128xf16>
    %empty = tensor.empty() : tensor<2x1026x128xf16>
    %result = linalg.generic {indexing_maps = [#id, #id],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%padded : tensor<2x1026x128xf16>)
        outs(%empty : tensor<2x1026x128xf16>) {
      ^bb0(%v: f16, %o: f16): linalg.yield %v : f16
    } -> tensor<2x1026x128xf16>
    return %result : tensor<2x1026x128xf16>
  }
}
)mlir");
  ASSERT_TRUE(support);
  auto supportDag =
      StructuredDAGAnalysis::create(function(*support), &failureReason);
  ASSERT_TRUE(mlir::succeeded(supportDag)) << failureReason;
  auto supportInputs = buildInputs(*supportDag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(supportInputs)) << failureReason;
  CanonicalStoragePlanOutcome supportOutcome = buildCanonicalStoragePlan(
      supportInputs->representations, supportInputs->movements,
      supportInputs->serialized);
  const CanonicalStorageCoordinate *supportCoordinate =
      getCanonicalStorageCoordinate(supportOutcome);
  ASSERT_NE(supportCoordinate, nullptr);
  unsigned supportObjects = 0;
  for (const StorageLifetimeDescription &lifetime :
       supportCoordinate->lifetimes) {
    const auto *version =
        std::get_if<PhysicalVersionId>(&lifetime.object.origin);
    if (!version ||
        !std::holds_alternative<SupportRegionValueId>(version->logicalValue))
      continue;
    ++supportObjects;
    ASSERT_EQ(lifetime.uses.size(), 1u);
    EXPECT_EQ(lifetime.definition, lifetime.uses.front());
  }
  EXPECT_EQ(supportObjects, 16u);
}

TEST_F(CanonicalStoragePlanTest,
       InvalidVersionsActionsResourcesAndLifetimesFailClosed) {
  auto module = parse(mapSource(1024));
  ASSERT_TRUE(module);
  std::string failureReason;
  auto dag = StructuredDAGAnalysis::create(function(*module), &failureReason);
  ASSERT_TRUE(mlir::succeeded(dag)) << failureReason;
  auto inputs = buildInputs(*dag, allTiles(), &failureReason);
  ASSERT_TRUE(mlir::succeeded(inputs)) << failureReason;

  CanonicalRepresentationCoordinate duplicateVersion = inputs->representations;
  duplicateVersion.resources.push_back(duplicateVersion.resources.front());
  CanonicalStoragePlanOutcome duplicateVersionOutcome =
      buildCanonicalStoragePlan(duplicateVersion, inputs->movements,
                                inputs->serialized);
  const auto *duplicateVersionFailure =
      std::get_if<BrokenStoragePlan>(&duplicateVersionOutcome);
  ASSERT_NE(duplicateVersionFailure, nullptr);
  EXPECT_EQ(duplicateVersionFailure->reason,
            BrokenStoragePlanReason::DuplicatePhysicalVersion);

  CanonicalRepresentationCoordinate missingVersion = inputs->representations;
  missingVersion.resources.pop_back();
  CanonicalStoragePlanOutcome missingVersionOutcome = buildCanonicalStoragePlan(
      missingVersion, inputs->movements, inputs->serialized);
  const auto *missingVersionFailure =
      std::get_if<BrokenStoragePlan>(&missingVersionOutcome);
  ASSERT_NE(missingVersionFailure, nullptr);
  EXPECT_EQ(missingVersionFailure->reason,
            BrokenStoragePlanReason::MissingPhysicalVersion);

  SerializedExecutionPlan missingExecution = inputs->serialized;
  missingExecution.executions.pop_back();
  CanonicalStoragePlanOutcome missingExecutionOutcome =
      buildCanonicalStoragePlan(inputs->representations, inputs->movements,
                                missingExecution);
  const auto *missingExecutionFailure =
      std::get_if<BrokenStoragePlan>(&missingExecutionOutcome);
  ASSERT_NE(missingExecutionFailure, nullptr);
  EXPECT_EQ(missingExecutionFailure->reason,
            BrokenStoragePlanReason::MissingSerializedExecution);

  CanonicalMovementCoordinate mismatchedBinding = inputs->movements;
  mismatchedBinding.plan.publications.front().source =
      mismatchedBinding.plan.externalLoads.front().destination;
  CanonicalStoragePlanOutcome mismatchedBindingOutcome =
      buildCanonicalStoragePlan(inputs->representations, mismatchedBinding,
                                inputs->serialized);
  const auto *mismatchedBindingFailure =
      std::get_if<BrokenStoragePlan>(&mismatchedBindingOutcome);
  ASSERT_NE(mismatchedBindingFailure, nullptr);
  EXPECT_EQ(mismatchedBindingFailure->reason,
            BrokenStoragePlanReason::PlanBindingMismatch);

  CanonicalMovementCoordinate duplicateAction = inputs->movements;
  duplicateAction.plan.externalLoads.push_back(
      duplicateAction.plan.externalLoads.front());
  CanonicalStoragePlanOutcome duplicateActionOutcome =
      buildCanonicalStoragePlan(inputs->representations, duplicateAction,
                                inputs->serialized);
  const auto *duplicateActionFailure =
      std::get_if<BrokenStoragePlan>(&duplicateActionOutcome);
  ASSERT_NE(duplicateActionFailure, nullptr);
  EXPECT_EQ(duplicateActionFailure->reason,
            BrokenStoragePlanReason::DuplicateMovementAction);

  ASSERT_FALSE(inputs->movements.plan.publications.empty());
  const ResultPublicationPlan &published =
      inputs->movements.plan.publications.front();
  ResultDiscardPlan discard{ResultDiscardId{published.id.source},
                            published.source};
  CanonicalMovementCoordinate duplicateDiscard = inputs->movements;
  duplicateDiscard.plan.discards = {discard, discard};
  CanonicalStoragePlanOutcome duplicateDiscardOutcome =
      buildCanonicalStoragePlan(inputs->representations, duplicateDiscard,
                                inputs->serialized);
  const auto *duplicateDiscardFailure =
      std::get_if<BrokenStoragePlan>(&duplicateDiscardOutcome);
  ASSERT_NE(duplicateDiscardFailure, nullptr);
  EXPECT_EQ(duplicateDiscardFailure->reason,
            BrokenStoragePlanReason::DuplicateResultDiscard);

  CanonicalMovementCoordinate carriedAndDiscarded = inputs->movements;
  carriedAndDiscarded.plan.discards.push_back(discard);
  CanonicalStoragePlanOutcome carriedAndDiscardedOutcome =
      buildCanonicalStoragePlan(inputs->representations, carriedAndDiscarded,
                                inputs->serialized);
  const auto *carriedAndDiscardedFailure =
      std::get_if<BrokenStoragePlan>(&carriedAndDiscardedOutcome);
  ASSERT_NE(carriedAndDiscardedFailure, nullptr);
  EXPECT_EQ(carriedAndDiscardedFailure->reason,
            BrokenStoragePlanReason::PlanBindingMismatch);

  CanonicalMovementCoordinate missingResource = inputs->movements;
  missingResource.resources.pop_back();
  CanonicalStoragePlanOutcome missingResourceOutcome =
      buildCanonicalStoragePlan(inputs->representations, missingResource,
                                inputs->serialized);
  const auto *missingResourceFailure =
      std::get_if<BrokenStoragePlan>(&missingResourceOutcome);
  ASSERT_NE(missingResourceFailure, nullptr);
  EXPECT_EQ(missingResourceFailure->reason,
            BrokenStoragePlanReason::MissingMovementResource);

  CanonicalMovementCoordinate mismatchedResource = inputs->movements;
  mismatchedResource.resources.front().elementType =
      mlir::Float32Type::get(context.get());
  CanonicalStoragePlanOutcome mismatchedResourceOutcome =
      buildCanonicalStoragePlan(inputs->representations, mismatchedResource,
                                inputs->serialized);
  const auto *mismatchedResourceFailure =
      std::get_if<BrokenStoragePlan>(&mismatchedResourceOutcome);
  ASSERT_NE(mismatchedResourceFailure, nullptr);
  EXPECT_EQ(mismatchedResourceFailure->reason,
            BrokenStoragePlanReason::ResourceMismatch);

  CanonicalMovementCoordinate missingDefinition = inputs->movements;
  MovementActionId loadAction = missingDefinition.plan.externalLoads.front().id;
  missingDefinition.plan.externalLoads.erase(
      missingDefinition.plan.externalLoads.begin());
  llvm::erase_if(missingDefinition.resources,
                 [&](const MovementResourceDescription &resource) {
                   return resource.action == loadAction;
                 });
  CanonicalStoragePlanOutcome missingDefinitionOutcome =
      buildCanonicalStoragePlan(inputs->representations, missingDefinition,
                                inputs->serialized);
  const auto *missingDefinitionFailure =
      std::get_if<BrokenStoragePlan>(&missingDefinitionOutcome);
  ASSERT_NE(missingDefinitionFailure, nullptr);
  EXPECT_EQ(missingDefinitionFailure->reason,
            BrokenStoragePlanReason::MissingDefinition);

  CanonicalMovementCoordinate missingUse = inputs->movements;
  MovementActionId publicationAction = missingUse.plan.publications.front().id;
  missingUse.plan.publications.erase(missingUse.plan.publications.begin());
  llvm::erase_if(missingUse.resources,
                 [&](const MovementResourceDescription &resource) {
                   return resource.action == publicationAction;
                 });
  CanonicalStoragePlanOutcome missingUseOutcome = buildCanonicalStoragePlan(
      inputs->representations, missingUse, inputs->serialized);
  const auto *missingUseFailure =
      std::get_if<BrokenStoragePlan>(&missingUseOutcome);
  ASSERT_NE(missingUseFailure, nullptr);
  EXPECT_EQ(missingUseFailure->reason, BrokenStoragePlanReason::MissingUse);

  CanonicalStoragePlanOutcome repeated = buildCanonicalStoragePlan(
      inputs->representations, inputs->movements, inputs->serialized);
  EXPECT_NE(getCanonicalStorageCoordinate(repeated), nullptr);
}

} // namespace
