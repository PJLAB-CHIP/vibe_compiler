//===- MovementTransferBuilderTest.cpp -------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/MovementTransferBuilder.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

#include <memory>

namespace {

using namespace wafer;
using namespace wafer::analysis;
using namespace wafer::compiler::detail;

ExactIndexSet makeExactDomain(llvm::ArrayRef<int64_t> offsets,
                              llvm::ArrayRef<int64_t> sizes) {
  IndexSetResult result =
      IndexRelation::staticRectangularDomain(offsets, sizes);
  EXPECT_TRUE(result.isExact()) << result.reason;
  return ExactIndexSet(
      std::move(*result.set), ExactIndexSetForm::BoxUnion,
      {StaticRectangularIndexSet{llvm::SmallVector<int64_t, 4>(offsets),
                                 llvm::SmallVector<int64_t, 4>(sizes)}});
}

ExactIndexSet makeExactUnion(llvm::ArrayRef<StaticRectangularIndexSet> boxes) {
  EXPECT_FALSE(boxes.empty());
  IndexSetResult first = IndexRelation::staticRectangularDomain(
      boxes.front().offsets, boxes.front().sizes);
  EXPECT_TRUE(first.isExact()) << first.reason;
  mlir::presburger::PresburgerSet combined = std::move(*first.set);
  for (const StaticRectangularIndexSet &box : boxes.drop_front()) {
    IndexSetResult next =
        IndexRelation::staticRectangularDomain(box.offsets, box.sizes);
    EXPECT_TRUE(next.isExact()) << next.reason;
    combined.unionInPlace(*next.set);
  }
  return ExactIndexSet(std::move(combined), ExactIndexSetForm::BoxUnion, boxes);
}

struct IRFixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  mlir::OpBuilder builder;
  llvm::SmallVector<mlir::Value, 4> buffers;
};

IRFixture makeIR(bool mismatchedLast = false, int64_t tileCount = 3,
                 int64_t major = 1025) {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::async::AsyncDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto function = moduleBuilder.create<mlir::func::FuncOp>(
      module.getLoc(), "movement",
      moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
  mlir::Block *entry = function.addEntryBlock();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(entry);
  llvm::SmallVector<mlir::Value, 4> buffers;
  for (int64_t tile = 0; tile < tileCount; ++tile) {
    MemLayout layout = mismatchedLast && tile + 1 == tileCount
                           ? MemLayout::NTensor
                           : MemLayout::Tensor;
    auto type = mlir::MemRefType::get(
        {2, major, 128}, builder.getF16Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(context.get(), MemorySpace::SPM, layout));
    buffers.push_back(
        builder.create<mlir::memref::AllocOp>(module.getLoc(), type));
  }
  return {std::move(context), mlir::OwningOpRef<mlir::ModuleOp>(module),
          function, builder, std::move(buffers)};
}

MovementActionId makeAction(int64_t destinationTile = 2,
                            uint64_t anchorIndex = 0) {
  SemanticRootKey root;
  root.anchorIndex = anchorIndex;
  RootRegionWorkId work{root, TileId(destinationTile)};
  LogicalShardId shard{root, {0}};
  DemandFragmentId fragment;
  fragment.source.kind = RootBoundaryKind::ProgramInput;
  fragment.use = {0, shard};
  return ExternalLoadId{BoundaryRegionValueId{work, fragment}};
}

MovementActionId makeBoundaryAction(int64_t destinationTile,
                                    uint64_t anchorIndex) {
  return DDRBoundaryTransferId{
      std::get<ExternalLoadId>(makeAction(destinationTile, anchorIndex))
          .destination};
}

void verifySelectedFanoutAtMajor(int64_t major) {
  IRFixture ir = makeIR(/*mismatchedLast=*/false, /*tileCount=*/4, major);
  PeerTransferGraphPlan realization;
  realization.kind = PeerTransferGraphKind::SoftwareFanout;
  realization.hops = {
      {TileId(0), TileId(1)}, {TileId(0), TileId(2)}, {TileId(1), TileId(3)}};
  realization.actions = {makeBoundaryAction(1, 1), makeBoundaryAction(2, 2),
                         makeBoundaryAction(3, 3)};
  llvm::sort(realization.actions);
  llvm::SmallVector<MovementEndpointBinding, 4> endpoints;
  for (int64_t tile = 0; tile < 4; ++tile)
    endpoints.push_back({TileId(tile), ir.buffers[tile], &ir.builder});
  MovementPlan plan;
  PhysicalVersionId source{RegionValueVersionId(
      std::get<DDRBoundaryTransferId>(realization.actions.front())
          .destination)};
  for (const MovementActionId &action : realization.actions) {
    DDRBoundaryTransferId transfer = std::get<DDRBoundaryTransferId>(action);
    plan.ddrTransfers.push_back(
        {transfer, source, PhysicalVersionId{transfer.destination}});
  }
  plan.peerGraphs.push_back(realization);
  llvm::sort(plan.ddrTransfers);
  llvm::SmallVector<int64_t, 4> offsets{0, 0, 0};
  llvm::SmallVector<int64_t, 4> sizes{2, major, 128};
  std::vector<MovementResourceDescription> resources;
  for (const MovementActionId &action : realization.actions)
    resources.push_back(
        {action, makeExactDomain(offsets, sizes), ir.builder.getF16Type(),
         TileId(0),
         std::get<DDRBoundaryTransferId>(action).destination.work.tile});
  std::string failureReason;
  SelectedPeerGraphBinding binding{
      realization.actions,
      {TileId(1), TileId(2), TileId(3)},
      std::vector<MovementEndpointBinding>(endpoints.begin(), endpoints.end())};
  binding.logicalOffsets = offsets;
  binding.logicalSizes = sizes;
  EXPECT_TRUE(mlir::failed(
      prepareSelectedPeerGraphs(plan, resources, {}, &failureReason)));
  unsigned preflightCommunicationOps = 0;
  ir.module->walk([&](CommPeerSendOp) { ++preflightCommunicationOps; });
  ir.module->walk([&](CommPeerRecvOp) { ++preflightCommunicationOps; });
  EXPECT_EQ(preflightCommunicationOps, 0u);
  auto prepared =
      prepareSelectedPeerGraphs(plan, resources, {binding}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  ASSERT_EQ(prepared->size(), 1u);
  EXPECT_EQ(prepared->front().transfer.actions, realization.actions);
  auto emitted = emitPreparedSelectedPeerGraphs(*prepared, &failureReason);
  ASSERT_TRUE(mlir::succeeded(emitted)) << failureReason;
  ASSERT_EQ(emitted->size(), 1u);
  EXPECT_EQ(emitted->front().transfer.hops.size(), 3u);
  EXPECT_EQ(emitted->front().transfer.sendTokens.size(), 3u);
  EXPECT_EQ(emitted->front().transfer.receiveTokens.size(), 3u);
  EXPECT_TRUE(mlir::succeeded(
      verifyTokenOnlySelectedPeerGraphs(*prepared, *emitted, &failureReason)))
      << failureReason;
  auto immediateAwait = ir.builder.create<mlir::async::AwaitOp>(
      ir.module->getLoc(), emitted->front().transfer.receiveTokens.front());
  EXPECT_TRUE(mlir::failed(
      verifyTokenOnlySelectedPeerGraphs(*prepared, *emitted, &failureReason)));
  immediateAwait.erase();
  ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
  unsigned awaits = 0;
  ir.module->walk([&](mlir::async::AwaitOp) { ++awaits; });
  EXPECT_EQ(awaits, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
}

TEST(MovementTransferBuilderTest,
     FanoutEmitsOneMatchingTokenPairPerGraphEdgeWithoutAwait) {
  verifySelectedFanoutAtMajor(1024);
  verifySelectedFanoutAtMajor(1025);
}

void verifyExternalFanoutAtMajor(int64_t major) {
  IRFixture ir = makeIR(/*mismatchedLast=*/false, /*tileCount=*/4, major);
  PeerTransferGraphPlan realization;
  realization.kind = PeerTransferGraphKind::ExternalLoadFanout;
  realization.hops = {{TileId(1), TileId(2)}, {TileId(1), TileId(3)}};
  realization.actions = {makeAction(1, 1), makeAction(2, 2), makeAction(3, 3)};
  llvm::sort(realization.actions);
  realization.ddrRoot = std::get<ExternalLoadId>(makeAction(1, 1));
  MovementPlan plan;
  for (const MovementActionId &action : realization.actions) {
    ExternalLoadId load = std::get<ExternalLoadId>(action);
    PhysicalVersionId destination{RegionValueVersionId(load.destination)};
    plan.externalLoads.push_back({load, destination});
  }
  plan.peerGraphs.push_back(realization);
  llvm::sort(plan.externalLoads);
  llvm::SmallVector<int64_t, 4> offsets{0, 0, 0};
  llvm::SmallVector<int64_t, 4> sizes{2, major, 128};
  std::vector<MovementResourceDescription> resources;
  for (const MovementActionId &action : realization.actions)
    resources.push_back(
        {action, makeExactDomain(offsets, sizes), ir.builder.getF16Type(),
         std::nullopt, std::get<ExternalLoadId>(action).destination.work.tile});
  auto ddrType = mlir::MemRefType::get(
      {2, major, 128}, ir.builder.getF16Type(),
      mlir::MemRefLayoutAttrInterface{},
      MemoryAttr::get(ir.context.get(), MemorySpace::DDR, MemLayout::Tensor));
  mlir::Value ddrSource =
      ir.builder.create<mlir::memref::AllocOp>(ir.module->getLoc(), ddrType);
  std::vector<MovementEndpointBinding> endpoints;
  for (int64_t tile = 0; tile < 4; ++tile)
    endpoints.push_back({TileId(tile), ir.buffers[tile], &ir.builder});
  SelectedPeerGraphBinding binding{realization.actions,
                                   {TileId(1), TileId(2), TileId(3)},
                                   endpoints,
                                   ddrSource};
  binding.logicalOffsets = offsets;
  binding.logicalSizes = sizes;
  std::string failureReason;
  auto prepared =
      prepareSelectedPeerGraphs(plan, resources, {binding}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto emitted = emitPreparedSelectedPeerGraphs(*prepared, &failureReason);
  ASSERT_TRUE(mlir::succeeded(emitted)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(
      verifyTokenOnlySelectedPeerGraphs(*prepared, *emitted, &failureReason)))
      << failureReason;
  unsigned loads = 0;
  unsigned sends = 0;
  ir.module->walk([&](StorageLoadOp) { ++loads; });
  ir.module->walk([&](CommPeerSendOp) { ++sends; });
  EXPECT_EQ(loads, 1u);
  EXPECT_EQ(sends, 2u);
  ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
}

TEST(MovementTransferBuilderTest,
     ExternalFanoutEmitsOneDDRRootAndTokenOnlyPeerTree) {
  verifyExternalFanoutAtMajor(1024);
  verifyExternalFanoutAtMajor(1025);
}

TEST(MovementTransferBuilderTest,
     RelayEmitsMatchingTokensWithoutImmediateAwait) {
  IRFixture ir = makeIR();
  PeerTransferGraphPlan realization;
  realization.kind = PeerTransferGraphKind::SoftwareRelay;
  realization.hops = {{TileId(0), TileId(1)}, {TileId(1), TileId(2)}};
  llvm::SmallVector<MovementEndpointBinding, 3> endpoints;
  for (int64_t tile = 0; tile < 3; ++tile)
    endpoints.push_back({TileId(tile), ir.buffers[tile], &ir.builder});
  std::string failureReason;
  auto prepared =
      preparePeerTransfer(makeAction(), realization, endpoints, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto emitted = emitPreparedPeerTransfer(
      *prepared, endpoints, /*communication=*/7, /*payload=*/3, &failureReason);
  ASSERT_TRUE(mlir::succeeded(emitted)) << failureReason;
  EXPECT_EQ(emitted->sendTokens.size(), 2u);
  EXPECT_EQ(emitted->receiveTokens.size(), 2u);
  ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
  unsigned sends = 0;
  unsigned receives = 0;
  unsigned awaits = 0;
  ir.module->walk([&](CommPeerSendOp) { ++sends; });
  ir.module->walk([&](CommPeerRecvOp) { ++receives; });
  ir.module->walk([&](mlir::async::AwaitOp) { ++awaits; });
  EXPECT_EQ(sends, 2u);
  EXPECT_EQ(receives, 2u);
  EXPECT_EQ(awaits, 0u);
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
}

TEST(MovementTransferBuilderTest,
     ReductionGatherUsesTheSameTokenOnlySelectedGraphBuilder) {
  IRFixture ir = makeIR(/*mismatchedLast=*/false, /*tileCount=*/3,
                        /*major=*/1031);
  SemanticRootKey root;
  RootRegionWorkId sourceWork{root, TileId(0)};
  RootRegionWorkId mergeWork{root, TileId(2)};
  LogicalShardId shard{root, {0}};
  ReductionGroupId group{root, 0, {0}};
  ExecutionInstanceId sourceExecution{RequiredRootExecution{sourceWork, shard}};
  ReductionPartialValueId logical{sourceExecution, group, 0};
  PhysicalVersionId source{RegionValueVersionId(logical)};
  ReductionGatherId gatherId{group, shard, logical};
  PeerTransferGraphPlan realization;
  realization.kind = PeerTransferGraphKind::SoftwareRelay;
  realization.hops = {{TileId(0), TileId(1)}, {TileId(1), TileId(2)}};
  realization.actions = {MovementActionId(gatherId)};
  MovementPlan plan;
  plan.reductionGathers.push_back(
      {gatherId, source,
       ExecutionInstanceId{RequiredMergeExecution{mergeWork, group}}});
  plan.peerGraphs.push_back(realization);
  llvm::SmallVector<int64_t, 4> offsets{0, 0, 0};
  llvm::SmallVector<int64_t, 4> sizes{2, 1031, 128};
  std::vector<MovementResourceDescription> resources{
      {MovementActionId(gatherId), makeExactDomain(offsets, sizes),
       ir.builder.getF16Type(), TileId(0), TileId(2)}};
  std::vector<MovementEndpointBinding> endpoints;
  for (int64_t tile = 0; tile < 3; ++tile)
    endpoints.push_back({TileId(tile), ir.buffers[tile], &ir.builder});
  SelectedPeerGraphBinding binding{realization.actions, {TileId(2)}, endpoints};
  binding.logicalOffsets = offsets;
  binding.logicalSizes = sizes;
  std::string failureReason;
  auto prepared =
      prepareSelectedPeerGraphs(plan, resources, {binding}, &failureReason);
  ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
  auto emitted = emitPreparedSelectedPeerGraphs(*prepared, &failureReason);
  ASSERT_TRUE(mlir::succeeded(emitted)) << failureReason;
  EXPECT_TRUE(mlir::succeeded(
      verifyTokenOnlySelectedPeerGraphs(*prepared, *emitted, &failureReason)))
      << failureReason;
  ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
}

TEST(MovementTransferBuilderTest,
     MultiPiecePayloadEmitsAllAndOnlyAlignedAndRaggedMessages) {
  for (int64_t major : {1024, 1025}) {
    IRFixture ir = makeIR(/*mismatchedLast=*/false, /*tileCount=*/3, major);
    MovementActionId action = makeBoundaryAction(2, 9);
    DDRBoundaryTransferId transfer = std::get<DDRBoundaryTransferId>(action);
    PhysicalVersionId source{RegionValueVersionId(transfer.destination)};
    MovementPlan plan;
    plan.ddrTransfers.push_back(
        {transfer, source, PhysicalVersionId{transfer.destination}});
    plan.peerGraphs.push_back({PeerTransferGraphKind::TargetRoutedPeer,
                               {{TileId(0), TileId(2)}},
                               {action}});
    const int64_t firstMajor = major / 2;
    std::vector<StaticRectangularIndexSet> boxes{
        {{0, 0, 0}, {2, firstMajor, 128}},
        {{0, firstMajor, 0}, {2, major - firstMajor, 128}}};
    std::vector<MovementResourceDescription> resources{
        {action, makeExactUnion(boxes), ir.builder.getF16Type(), TileId(0),
         TileId(2)}};
    std::vector<SelectedPeerGraphBinding> bindings;
    for (auto [slice, box] : llvm::enumerate(boxes)) {
      std::vector<MovementEndpointBinding> endpoints;
      for (TileId tile : {TileId(0), TileId(2)}) {
        auto type = mlir::MemRefType::get(box.sizes, ir.builder.getF16Type(),
                                          mlir::MemRefLayoutAttrInterface{},
                                          MemoryAttr::get(ir.context.get(),
                                                          MemorySpace::SPM,
                                                          MemLayout::Tensor));
        mlir::Value buffer =
            ir.builder.create<mlir::memref::AllocOp>(ir.module->getLoc(), type);
        endpoints.push_back({tile, buffer, &ir.builder});
      }
      SelectedPeerGraphBinding binding{{action}, {TileId(2)}, endpoints};
      binding.payloadSlice = static_cast<uint32_t>(slice);
      binding.logicalOffsets = box.offsets;
      binding.logicalSizes = box.sizes;
      bindings.push_back(std::move(binding));
    }
    std::string failureReason;
    EXPECT_TRUE(mlir::failed(prepareSelectedPeerGraphs(
        plan, resources,
        llvm::ArrayRef<SelectedPeerGraphBinding>(bindings).drop_back(),
        &failureReason)));
    unsigned sendsBefore = 0;
    ir.module->walk([&](CommPeerSendOp) { ++sendsBefore; });
    EXPECT_EQ(sendsBefore, 0u);
    auto prepared =
        prepareSelectedPeerGraphs(plan, resources, bindings, &failureReason);
    ASSERT_TRUE(mlir::succeeded(prepared)) << failureReason;
    ASSERT_EQ(prepared->size(), 2u);
    EXPECT_EQ(prepared->front().communication, prepared->back().communication);
    EXPECT_EQ(prepared->front().payload, 0);
    EXPECT_EQ(prepared->back().payload, 1);
    auto emitted = emitPreparedSelectedPeerGraphs(*prepared, &failureReason);
    ASSERT_TRUE(mlir::succeeded(emitted)) << failureReason;
    EXPECT_TRUE(mlir::succeeded(
        verifyTokenOnlySelectedPeerGraphs(*prepared, *emitted, &failureReason)))
        << failureReason;
    unsigned sends = 0;
    ir.module->walk([&](CommPeerSendOp) { ++sends; });
    EXPECT_EQ(sends, 2u);
    ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
  }
}

TEST(MovementTransferBuilderTest,
     PhysicalMismatchAndCycleFailBeforeAnyCommunicationMutation) {
  IRFixture ir = makeIR(/*mismatchedLast=*/true);
  PeerTransferGraphPlan realization;
  realization.kind = PeerTransferGraphKind::SoftwareRelay;
  realization.hops = {{TileId(0), TileId(1)}, {TileId(1), TileId(2)}};
  llvm::SmallVector<MovementEndpointBinding, 3> endpoints;
  for (int64_t tile = 0; tile < 3; ++tile)
    endpoints.push_back({TileId(tile), ir.buffers[tile], &ir.builder});
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(preparePeerTransfer(makeAction(), realization,
                                               endpoints, &failureReason)));
  unsigned communicationOps = 0;
  ir.module->walk([&](CommPeerSendOp) { ++communicationOps; });
  ir.module->walk([&](CommPeerRecvOp) { ++communicationOps; });
  EXPECT_EQ(communicationOps, 0u);

  realization.hops = {{TileId(0), TileId(1)}, {TileId(1), TileId(0)}};
  EXPECT_TRUE(mlir::failed(preparePeerTransfer(makeAction(), realization,
                                               endpoints, &failureReason)));
  EXPECT_EQ(communicationOps, 0u);
}

} // namespace
