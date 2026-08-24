//===- StorageDomainTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/StorageDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <set>
#include <vector>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExactIndexSet makeDomain(llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<int64_t, 4> offsets(sizes.size(), 0);
  IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(set.isExact()) << set.reason;
  StaticRectangularIndexSet box{offsets, llvm::SmallVector<int64_t, 4>(sizes)};
  return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion, {box});
}

PhysicalVersionId makeVersion(uint32_t result) {
  SemanticRootKey root;
  RootRegionWorkId work{root, TileId(0)};
  LogicalShardId shard{root, {0}};
  return PhysicalVersionId{ExecutionResultValueId{
      ExecutionInstanceId{RequiredRootExecution{work, shard}}, result}};
}

CanonicalStorageCoordinate makeStorage(mlir::MLIRContext &context,
                                       unsigned count,
                                       llvm::ArrayRef<int64_t> sizes) {
  CanonicalStorageCoordinate coordinate;
  for (unsigned index = 0; index < count; ++index) {
    PhysicalVersionId version = makeVersion(index);
    StorageObjectId object{StorageObjectOrigin(version)};
    coordinate.plan.storageObjects.push_back({object, TileId(0)});
    coordinate.plan.versionBindings.push_back(
        {version, object, StorageBindingKind::Fresh});
    coordinate.resources.emplace_back(object, makeDomain(sizes),
                                      mlir::Float16Type::get(&context),
                                      MemLayout::Tensor);
  }
  return coordinate;
}

std::optional<std::vector<BufferPlan>> enumerate(const StorageDomain &domain,
                                                 size_t limit = 10000) {
  std::vector<BufferPlan> plans;
  StorageSuccessor next = domain.getFirstPlan();
  while (next.getKind() == StorageSuccessorKind::Plan) {
    if (!next.getPlan() || !next.getCursor() || plans.size() >= limit ||
        !domain.contains(*next.getPlan()))
      return std::nullopt;
    plans.push_back(*next.getPlan());
    StorageCursor cursor = *next.getCursor();
    next = domain.getNextPlan(cursor);
  }
  return next.getKind() == StorageSuccessorKind::End
             ? std::optional<std::vector<BufferPlan>>(std::move(plans))
             : std::nullopt;
}

TEST(StorageDomainTest,
     FreshReuseAndEverySlotMultiplicityMatchIndependentProduct) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  CanonicalStorageCoordinate storage = makeStorage(context, 3, {2, 1025, 128});
  PhysicalVersionId first = makeVersion(0);
  PhysicalVersionId second = makeVersion(1);
  PhysicalVersionId third = makeVersion(2);
  StorageObjectId firstObject{StorageObjectOrigin(first)};
  StorageObjectId secondObject{StorageObjectOrigin(second)};
  std::vector<StorageReuseRequirement> reuse{
      {second, firstObject, StorageReuseProof::ProvenDisjoint},
      {third, secondObject, StorageReuseProof::RequiresOrder}};
  SlotFamilyRequirement family;
  family.id.objects = {StorageObjectId{StorageObjectOrigin(first)},
                       StorageObjectId{StorageObjectOrigin(second)}};
  family.occurrence = OccurrenceRelationId{
      TraversalScopeId{
          RegionExecutionId{
              std::get<ExecutionResultValueId>(first.logicalValue).execution},
          TopLevelWorkPieceId{0}},
      {2, 5, 1}};
  family.upperBound = 5;
  family.rotationOptions = {{0, 1}, {1, 0}};
  StorageDomainResult result = buildStorageDomain(storage, reuse, {family});
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  EXPECT_EQ(plans->size(), 2u * 2u * (1u + 4u * 2u));

  std::set<uint32_t> multiplicities;
  bool sawFresh = false;
  bool sawReuse = false;
  bool sawOrderedReuse = false;
  for (const BufferPlan &plan : *plans) {
    ASSERT_EQ(plan.slotFamilies.size(), 1u);
    multiplicities.insert(plan.slotFamilies.front().multiplicity);
    sawFresh |= plan.storageObjects.size() == 3;
    sawReuse |= llvm::any_of(plan.versionBindings, [](const auto &binding) {
      return binding.kind == StorageBindingKind::Reuse;
    });
    sawOrderedReuse |= !plan.orderRequirements.empty();
  }
  EXPECT_EQ(multiplicities, (std::set<uint32_t>{1, 2, 3, 4, 5}));
  EXPECT_TRUE(sawFresh);
  EXPECT_TRUE(sawReuse);
  EXPECT_TRUE(sawOrderedReuse);
}

TEST(StorageDomainTest, IdentityAliasUsesSourceObjectWithoutFreshAllocation) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  CanonicalStorageCoordinate storage = makeStorage(context, 2, {2, 1031, 128});
  PhysicalVersionId source = makeVersion(0);
  PhysicalVersionId alias = makeVersion(1);
  alias.derivation.push_back(
      {PhysicalVersionDerivationKind::AliasView, MemLayout::Tensor,
       MemLayout::Tensor, SharedRepresentationAnchor{}, source.logicalValue});
  StorageObjectId oldAliasObject = storage.plan.storageObjects[1].id;
  storage.plan.storageObjects[1].id =
      StorageObjectId{StorageObjectOrigin(alias)};
  storage.plan.versionBindings[1] = {alias, storage.plan.storageObjects[1].id,
                                     StorageBindingKind::Fresh};
  storage.resources[1].object = storage.plan.storageObjects[1].id;
  (void)oldAliasObject;
  StorageDomainResult result = buildStorageDomain(storage);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  StorageSuccessor first = result.domain->getFirstPlan();
  ASSERT_NE(first.getPlan(), nullptr);
  ASSERT_EQ(first.getPlan()->storageObjects.size(), 1u);
  auto binding = llvm::find_if(
      first.getPlan()->versionBindings,
      [&](const auto &candidate) { return candidate.version == alias; });
  ASSERT_NE(binding, first.getPlan()->versionBindings.end());
  EXPECT_EQ(binding->kind, StorageBindingKind::IdentityAlias);
  EXPECT_EQ(binding->object, StorageObjectId{StorageObjectOrigin(source)});
}

TEST(StorageDomainTest, OrderedReuseCycleIsExcludedButFreshSiblingsRemain) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  CanonicalStorageCoordinate storage = makeStorage(context, 2, {2, 1025, 128});
  PhysicalVersionId first = makeVersion(0);
  PhysicalVersionId second = makeVersion(1);
  std::vector<StorageReuseRequirement> reuse{
      {first, StorageObjectId{StorageObjectOrigin(second)},
       StorageReuseProof::RequiresOrder},
      {second, StorageObjectId{StorageObjectOrigin(first)},
       StorageReuseProof::RequiresOrder}};
  StorageDomainResult result = buildStorageDomain(storage, reuse);
  ASSERT_TRUE(result.succeeded())
      << (result.failure ? result.failure->detail : "");
  auto plans = enumerate(*result.domain);
  ASSERT_TRUE(plans);
  EXPECT_EQ(plans->size(), 3u);
  EXPECT_TRUE(llvm::all_of(*plans, [](const BufferPlan &plan) {
    return plan.orderRequirements.size() < 2;
  }));
}

TEST(StorageDomainTest, IncompatibleReuseAndInvalidSlotUpperFailTyped) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  CanonicalStorageCoordinate storage = makeStorage(context, 2, {2, 1025, 128});
  storage.resources[1] = StorageResourceDescription{
      storage.resources[1].object, makeDomain({2, 1024, 128}),
      mlir::Float16Type::get(&context), MemLayout::Tensor};
  StorageReuseRequirement reuse{
      makeVersion(1), StorageObjectId{StorageObjectOrigin(makeVersion(0))},
      StorageReuseProof::ProvenDisjoint};
  StorageDomainResult unsupported = buildStorageDomain(storage, {reuse});
  ASSERT_FALSE(unsupported.succeeded());
  ASSERT_TRUE(unsupported.failure);
  EXPECT_EQ(unsupported.failure->kind,
            StorageDomainFailureKind::UnsupportedSemantics);

  SlotFamilyRequirement invalid;
  invalid.id.objects = {StorageObjectId{StorageObjectOrigin(makeVersion(0))}};
  invalid.upperBound = 0;
  StorageDomainResult broken = buildStorageDomain(storage, {}, {invalid});
  ASSERT_FALSE(broken.succeeded());
  ASSERT_TRUE(broken.failure);
  EXPECT_EQ(broken.failure->kind, StorageDomainFailureKind::BrokenContract);

  SlotFamilyRequirement overTrip;
  overTrip.id.objects = {StorageObjectId{StorageObjectOrigin(makeVersion(0))}};
  overTrip.occurrence = OccurrenceRelationId{TraversalScopeId{}, {2, 1, 1}};
  overTrip.upperBound = 3;
  overTrip.rotationOptions = {{0}};
  StorageDomainResult impossibleMultiplicity =
      buildStorageDomain(storage, {}, {overTrip});
  ASSERT_FALSE(impossibleMultiplicity.succeeded());
  ASSERT_TRUE(impossibleMultiplicity.failure);
  EXPECT_EQ(impossibleMultiplicity.failure->kind,
            StorageDomainFailureKind::BrokenContract);

  StorageDomainResult duplicateReuse =
      buildStorageDomain(storage, {reuse, reuse});
  ASSERT_FALSE(duplicateReuse.succeeded());
  ASSERT_TRUE(duplicateReuse.failure);
  EXPECT_EQ(duplicateReuse.failure->kind,
            StorageDomainFailureKind::BrokenContract);
}

} // namespace
