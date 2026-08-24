//===- StorageRequirementsTest.cpp ----------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/StorageRequirements.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"

#include "gtest/gtest.h"

#include <set>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExactIndexSet makeDomain(llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<int64_t, 4> offsets(sizes.size(), 0);
  IndexSetResult set = IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(set.isExact()) << set.reason;
  return ExactIndexSet(std::move(*set.set), ExactIndexSetForm::BoxUnion,
                       {StaticRectangularIndexSet{
                           offsets, llvm::SmallVector<int64_t, 4>(sizes)}});
}

PhysicalVersionId makeVersion(uint64_t anchor, uint32_t result = 0,
                              TileId tile = TileId(0)) {
  SemanticRootKey root;
  root.anchorIndex = anchor;
  RootRegionWorkId work{root, tile};
  LogicalShardId shard{root, {0}};
  return PhysicalVersionId{ExecutionResultValueId{
      ExecutionInstanceId{RequiredRootExecution{work, shard}}, result}};
}

ExecutionInstanceId getExecution(const PhysicalVersionId &version) {
  return std::get<ExecutionInstanceId>(
      std::get<ExecutionResultValueId>(version.logicalValue).execution);
}

void addVersion(CanonicalStorageCoordinate &storage,
                RepresentationPlan &representations,
                const PhysicalVersionId &version, const ExactIndexSet &domain,
                mlir::Type elementType) {
  StorageObjectId object{StorageObjectOrigin(version)};
  storage.plan.storageObjects.push_back({object, TileId(0)});
  storage.plan.versionBindings.push_back(
      {version, object, StorageBindingKind::Fresh});
  storage.resources.emplace_back(object, domain, elementType,
                                 MemLayout::Tensor);
  storage.lifetimes.push_back({object,
                               StorageAccessSite{getExecution(version)},
                               {StorageAccessSite{getExecution(version)}}});
  representations.logicalValues.push_back({version.logicalValue, version});
  representations.physicalVersions.push_back({version, MemLayout::Tensor});
}

TEST(StorageRequirementsTest,
     TemporalOccurrencesDeriveEveryMultiplicityWithoutCapacityInput) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (int64_t major : {1024, 1025}) {
    CanonicalStorageCoordinate storage;
    RepresentationPlan representations;
    PhysicalVersionId first = makeVersion(0, 0);
    PhysicalVersionId second = makeVersion(0, 1);
    PhysicalVersionId third = makeVersion(0, 2);
    for (const PhysicalVersionId &version : {first, second, third})
      addVersion(storage, representations, version, makeDomain({2, major, 128}),
                 mlir::Float16Type::get(&context));
    llvm::sort(storage.plan.storageObjects);
    llvm::sort(storage.plan.versionBindings);
    llvm::sort(storage.resources, [](const auto &lhs, const auto &rhs) {
      return lhs.object < rhs.object;
    });
    llvm::sort(storage.lifetimes, [](const auto &lhs, const auto &rhs) {
      return lhs.object < rhs.object;
    });
    llvm::sort(representations.logicalValues);
    llvm::sort(representations.physicalVersions);

    TraversalScopeId scope{RegionExecutionId{getExecution(first)},
                           TopLevelWorkPieceId{0}};
    TemporalPlan temporal{{{scope, {1, 256, 128}, {0, 1}}}};
    TemporalScopeDescriptor descriptor{scope,
                                       {0, 0, 0},
                                       {2, major, 128},
                                       {IteratorTilingCapability::Tileable,
                                        IteratorTilingCapability::Tileable,
                                        IteratorTilingCapability::Tileable},
                                       {},
                                       std::nullopt};
    StorageRequirementDerivationResult derived = deriveStorageRequirements(
        storage, representations, MovementPlan{}, temporal, {descriptor});
    ASSERT_TRUE(derived.succeeded())
        << (derived.failure ? derived.failure->detail : "");
    ASSERT_EQ(derived.requirements->slotFamilies.size(), 1u);
    const SlotFamilyRequirement &family =
        derived.requirements->slotFamilies.front();
    EXPECT_EQ(family.id.objects.size(), 3u);
    EXPECT_EQ(family.occurrence.axisOccurrences,
              (llvm::SmallVector<uint64_t, 4>{
                  2, static_cast<uint64_t>((major + 255) / 256), 1}));
    EXPECT_EQ(family.upperBound,
              static_cast<uint32_t>(2 * ((major + 255) / 256)));
    EXPECT_EQ(family.rotationOptions.size(), 2u);

    CanonicalStorageCoordinate reversedStorage = storage;
    std::reverse(reversedStorage.plan.storageObjects.begin(),
                 reversedStorage.plan.storageObjects.end());
    std::reverse(reversedStorage.plan.versionBindings.begin(),
                 reversedStorage.plan.versionBindings.end());
    std::reverse(reversedStorage.resources.begin(),
                 reversedStorage.resources.end());
    std::reverse(reversedStorage.lifetimes.begin(),
                 reversedStorage.lifetimes.end());
    RepresentationPlan reversedRepresentations = representations;
    std::reverse(reversedRepresentations.logicalValues.begin(),
                 reversedRepresentations.logicalValues.end());
    std::reverse(reversedRepresentations.physicalVersions.begin(),
                 reversedRepresentations.physicalVersions.end());
    StorageRequirementDerivationResult reordered =
        deriveStorageRequirements(reversedStorage, reversedRepresentations,
                                  MovementPlan{}, temporal, {descriptor});
    ASSERT_TRUE(reordered.succeeded());
    EXPECT_EQ(reordered.requirements->reuse, derived.requirements->reuse);
    EXPECT_EQ(reordered.requirements->slotFamilies,
              derived.requirements->slotFamilies);

    CanonicalStorageCoordinate resized = storage;
    for (StorageResourceDescription &resource : resized.resources)
      resource = StorageResourceDescription{
          resource.object, makeDomain({2, major * 2, 128}),
          mlir::Float16Type::get(&context), MemLayout::Tensor};
    StorageRequirementDerivationResult byteIndependent =
        deriveStorageRequirements(resized, representations, MovementPlan{},
                                  temporal, {descriptor});
    ASSERT_TRUE(byteIndependent.succeeded());
    EXPECT_EQ(byteIndependent.requirements->slotFamilies,
              derived.requirements->slotFamilies);

    StorageDomainResult domain =
        buildStorageDomain(storage, derived.requirements->reuse,
                           derived.requirements->slotFamilies);
    ASSERT_TRUE(domain.succeeded())
        << (domain.failure ? domain.failure->detail : "");
    std::set<uint32_t> multiplicities;
    StorageSuccessor next = domain.domain->getFirstPlan();
    while (next.getKind() == StorageSuccessorKind::Plan) {
      ASSERT_NE(next.getPlan(), nullptr);
      multiplicities.insert(next.getPlan()->slotFamilies.front().multiplicity);
      StorageCursor cursor = *next.getCursor();
      next = domain.domain->getNextPlan(cursor);
    }
    EXPECT_EQ(multiplicities.size(), family.upperBound);
    EXPECT_EQ(*multiplicities.begin(), 1u);
    EXPECT_EQ(*multiplicities.rbegin(), family.upperBound);

    if (major == 1024) {
      StorageRequirementDerivationResult exhausted = deriveStorageRequirements(
          storage, representations, MovementPlan{}, temporal, {descriptor},
          StorageRequirementLimits{/*maxRotationOptions=*/1});
      ASSERT_FALSE(exhausted.succeeded());
      ASSERT_TRUE(exhausted.failure);
      EXPECT_EQ(exhausted.failure->kind,
                StorageRequirementFailureKind::Indeterminate);
    }
  }
}

TEST(StorageRequirementsTest,
     CompatibleVersionsProduceOrderConstrainedReuseButForcedOverlapDoesNot) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  CanonicalStorageCoordinate storage;
  RepresentationPlan representations;
  for (uint64_t anchor = 0; anchor < 3; ++anchor)
    addVersion(storage, representations, makeVersion(anchor),
               makeDomain({2, 1031, 128}), mlir::Float16Type::get(&context));
  StorageRequirementDerivationResult derived = deriveStorageRequirements(
      storage, representations, MovementPlan{}, TemporalPlan{}, {});
  ASSERT_TRUE(derived.succeeded());
  EXPECT_EQ(derived.requirements->reuse.size(), 6u);
  EXPECT_TRUE(llvm::all_of(derived.requirements->reuse, [](const auto &reuse) {
    return reuse.proof == StorageReuseProof::RequiresOrder;
  }));

  storage.lifetimes[1].uses.push_back(storage.lifetimes[0].definition);
  StorageRequirementDerivationResult overlap = deriveStorageRequirements(
      storage, representations, MovementPlan{}, TemporalPlan{}, {});
  ASSERT_TRUE(overlap.succeeded());
  EXPECT_LT(overlap.requirements->reuse.size(),
            derived.requirements->reuse.size());
}

TEST(StorageRequirementsTest,
     PeerRelayObjectUsesTheConsumerExactOccurrenceClass) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  PhysicalVersionId source = makeVersion(0, 0, TileId(0));
  SemanticRootKey destinationRoot;
  destinationRoot.anchorIndex = 1;
  RootRegionWorkId destinationWork{destinationRoot, TileId(2)};
  LogicalShardId destinationShard{destinationRoot, {0}};
  DemandFragmentId fragment;
  fragment.source.kind = RootBoundaryKind::StructuredResult;
  const auto &sourceRoot =
      std::get<RequiredRootExecution>(getExecution(source).source);
  fragment.source.semantic = sourceRoot.work.root;
  fragment.ownerShard = sourceRoot.shard;
  fragment.ownerTile = TileId(0);
  fragment.use = {0, destinationShard};
  BoundaryRegionValueId destination{destinationWork, fragment};
  DDRBoundaryTransferId transfer{destination};
  MovementPlan movement;
  movement.ddrTransfers.push_back(
      {transfer, source, PhysicalVersionId{destination}});
  movement.peerGraphs.push_back(
      {PeerTransferGraphKind::SoftwareRelay,
       {{TileId(0), TileId(1)}, {TileId(1), TileId(2)}},
       {MovementActionId(transfer)}});

  CanonicalStorageCoordinate storage;
  RepresentationPlan representations;
  addVersion(storage, representations, source, makeDomain({2, 1025, 128}),
             mlir::Float16Type::get(&context));
  PeerRelayStorageId relay{{MovementActionId(transfer)}, 0, TileId(1)};
  StorageObjectId relayObject{StorageObjectOrigin(relay)};
  storage.plan.storageObjects.push_back({relayObject, TileId(1)});
  storage.resources.emplace_back(relayObject, makeDomain({2, 1025, 128}),
                                 mlir::Float16Type::get(&context),
                                 MemLayout::Tensor);
  PeerTransferSiteId receive{relay.graphActions,
                             0,
                             PeerTransferSiteId::Endpoint::Receive,
                             {TileId(0), TileId(1)}};
  PeerTransferSiteId send{relay.graphActions,
                          0,
                          PeerTransferSiteId::Endpoint::Send,
                          {TileId(1), TileId(2)}};
  storage.lifetimes.push_back(
      {relayObject, StorageAccessSite(receive), {StorageAccessSite(send)}});

  ExecutionInstanceId consumer{
      RequiredRootExecution{destinationWork, destinationShard}};
  TraversalScopeId scope{RegionExecutionId{consumer}, TopLevelWorkPieceId{0}};
  TemporalPlan temporal{{{scope, {1, 256, 128}, {0, 1}}}};
  TemporalScopeDescriptor descriptor{scope,
                                     {0, 0, 0},
                                     {2, 1025, 128},
                                     {IteratorTilingCapability::Tileable,
                                      IteratorTilingCapability::Tileable,
                                      IteratorTilingCapability::Tileable},
                                     {},
                                     std::nullopt};
  StorageRequirementDerivationResult derived = deriveStorageRequirements(
      storage, representations, movement, temporal, {descriptor});
  ASSERT_TRUE(derived.succeeded())
      << (derived.failure ? derived.failure->detail : "");
  ASSERT_EQ(derived.requirements->slotFamilies.size(), 1u);
  EXPECT_TRUE(llvm::is_contained(
      derived.requirements->slotFamilies.front().id.objects, relayObject));
}

} // namespace
