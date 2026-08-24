//===- StorageObjectBuilderTest.cpp ----------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/StorageObjectBuilder.h"
#include "Wafer/Planning/PhysicalDataflow/ExecutionStructureMaterialization.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/InitWaferDialects.h"
#include "Wafer/Transforms/MemoryPlanning.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

#include "gtest/gtest.h"

#include <memory>

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

ExecutionInstanceId makeExecution(uint32_t result) {
  return std::get<ExecutionInstanceId>(
      std::get<ExecutionResultValueId>(makeVersion(result).logicalValue)
          .execution);
}

struct IRFixture {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp function;
  mlir::OpBuilder builder;
};

IRFixture makeIR() {
  mlir::DialectRegistry registry;
  registerWaferCoreDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::async::AsyncDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  mlir::OpBuilder moduleBuilder(module.getBodyRegion());
  auto function = moduleBuilder.create<mlir::func::FuncOp>(
      module.getLoc(), "storage",
      moduleBuilder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{}));
  mlir::Block *entry = function.addEntryBlock();
  return {std::move(context), mlir::OwningOpRef<mlir::ModuleOp>(module),
          function, mlir::OpBuilder::atBlockBegin(entry)};
}

TEST(StorageObjectBuilderTest,
     MultiSlotAndIdentityAliasUseExactSelectedObjects) {
  for (uint32_t multiplicity : {2u, 3u, 4u, 5u}) {
    IRFixture ir = makeIR();
    PhysicalVersionId source = makeVersion(0);
    PhysicalVersionId alias = makeVersion(1);
    StorageObjectId object{StorageObjectOrigin(source)};
    PreparedStoragePlan prepared;
    prepared.objects.push_back(
        {{object, TileId(0)},
         StorageResourceDescription{object, makeDomain({2, 1025, 128}),
                                    ir.builder.getF16Type(), MemLayout::Tensor},
         multiplicity});
    prepared.bindings.push_back({source, object, StorageBindingKind::Fresh});
    prepared.bindings.push_back(
        {alias, object, StorageBindingKind::IdentityAlias});
    OccurrenceRelationId occurrence;
    occurrence.axisOccurrences = {2, 5, 1};
    prepared.slotFamilies.push_back(
        {SlotFamilyId{{object}}, occurrence, multiplicity, {0, 1}});
    StorageObjectBuilder objects;
    std::string failureReason;
    ASSERT_TRUE(mlir::succeeded(emitPreparedStorageObjects(
        prepared, objects, ir.builder, &failureReason)))
        << failureReason;
    EXPECT_TRUE(mlir::succeeded(
        verifyEmittedStorageObjects(prepared, objects, &failureReason)))
        << failureReason;
    unsigned allocations = 0;
    ir.module->walk([&](mlir::memref::AllocOp) { ++allocations; });
    EXPECT_EQ(allocations, multiplicity);
    EXPECT_EQ(objects.lookup(source, 0), objects.lookup(source, multiplicity));
    EXPECT_NE(objects.lookup(source, 0), objects.lookup(source, 1));
    EXPECT_EQ(objects.lookup(alias, multiplicity + 2),
              objects.lookup(source, multiplicity + 2));
    EXPECT_EQ(objects.lookup(object, multiplicity * 2 + 1),
              objects.lookup(source, 1));
    auto coordinate = objects.lookup(source, {1, 2, 0});
    ASSERT_TRUE(mlir::succeeded(coordinate));
    EXPECT_EQ(*coordinate, objects.lookup(source, (5 + 2) % multiplicity));
    auto aliasCoordinate = objects.lookup(alias, {1, 2, 0});
    ASSERT_TRUE(mlir::succeeded(aliasCoordinate));
    EXPECT_EQ(*aliasCoordinate, *coordinate);
    EXPECT_TRUE(mlir::failed(objects.lookup(source, {2, 0, 0})));
  }
}

TEST(StorageObjectBuilderTest,
     DuplicatePreparedObjectFailsBeforeAnyAllocationMutation) {
  IRFixture ir = makeIR();
  PhysicalVersionId version = makeVersion(0);
  StorageObjectId object{StorageObjectOrigin(version)};
  PreparedStorageObject preparedObject{
      {object, TileId(0)},
      StorageResourceDescription{object, makeDomain({2, 1031, 128}),
                                 ir.builder.getF16Type(), MemLayout::Tensor},
      1};
  PreparedStoragePlan prepared;
  prepared.objects = {preparedObject, preparedObject};
  prepared.bindings.push_back({version, object, StorageBindingKind::Fresh});
  StorageObjectBuilder objects;
  std::string failureReason;
  EXPECT_TRUE(mlir::failed(emitPreparedStorageObjects(
      prepared, objects, ir.builder, &failureReason)));
  unsigned allocations = 0;
  ir.module->walk([&](mlir::memref::AllocOp) { ++allocations; });
  EXPECT_EQ(allocations, 0u);
}

TEST(StorageObjectBuilderTest,
     PeerRelaySlotsUseTypedObjectIdentityAndExactModuloLookup) {
  IRFixture ir = makeIR();
  MovementActionId action = ExternalLoadId{};
  StorageObjectId relay{
      StorageObjectOrigin{PeerRelayStorageId{{action}, 0, TileId(3)}}};
  PreparedStoragePlan prepared;
  prepared.objects.push_back(
      {{relay, TileId(3)},
       StorageResourceDescription{relay, makeDomain({2, 1024, 128}),
                                  ir.builder.getF16Type(), MemLayout::Tensor},
       2});
  OccurrenceRelationId occurrence;
  occurrence.axisOccurrences = {2, 1, 1};
  prepared.slotFamilies.push_back({SlotFamilyId{{relay}}, occurrence, 2, {0}});
  StorageObjectBuilder objects;
  std::string failureReason;
  ASSERT_TRUE(mlir::succeeded(emitPreparedStorageObjects(
      prepared, objects, ir.builder, &failureReason)))
      << failureReason;
  ASSERT_TRUE(mlir::succeeded(
      verifyEmittedStorageObjects(prepared, objects, &failureReason)))
      << failureReason;
  EXPECT_NE(objects.lookup(relay, 0), objects.lookup(relay, 1));
  EXPECT_EQ(objects.lookup(relay, 0), objects.lookup(relay, 2));
}

TEST(StorageObjectBuilderTest,
     ActualAsyncLifetimeRequiresCompletionBeforeSlotRelease) {
  for (int64_t major : {1024, 1025}) {
    IRFixture ir = makeIR();
    PhysicalVersionId version = makeVersion(0);
    StorageObjectId object{StorageObjectOrigin(version)};
    PreparedStoragePlan prepared;
    prepared.objects.push_back(
        {{object, TileId(0)},
         StorageResourceDescription{object, makeDomain({2, major, 128}),
                                    ir.builder.getF16Type(), MemLayout::Tensor},
         2});
    prepared.bindings.push_back({version, object, StorageBindingKind::Fresh});
    OccurrenceRelationId occurrence;
    occurrence.axisOccurrences = {2, 1, 1};
    prepared.slotFamilies.push_back(
        {SlotFamilyId{{object}}, occurrence, 2, {0}});
    StorageObjectBuilder objects;
    std::string failureReason;
    ASSERT_TRUE(mlir::succeeded(emitPreparedStorageObjects(
        prepared, objects, ir.builder, &failureReason)))
        << failureReason;
    auto ddrType = mlir::MemRefType::get(
        {2, major, 128}, ir.builder.getF16Type(),
        mlir::MemRefLayoutAttrInterface{},
        MemoryAttr::get(ir.context.get(), MemorySpace::DDR, MemLayout::Tensor));
    mlir::Value ddr =
        ir.builder.create<mlir::memref::AllocOp>(ir.module->getLoc(), ddrType);
    std::vector<SelectedStorageLifetimeBinding> lifetimes;
    const int64_t bytes = 2 * major * 128 * 2;
    for (uint64_t wave = 0; wave < 2; ++wave) {
      mlir::Value buffer = objects.lookup(object, wave);
      auto definition =
          ir.builder.create<StorageLoadOp>(ir.module->getLoc(), ddr, buffer);
      auto message = DTEMessageAttr::get(ir.context.get(), 0, 0, wave);
      auto send = ir.builder.create<CommPeerSendOp>(
          ir.module->getLoc(), mlir::async::TokenType::get(ir.context.get()),
          buffer, ir.builder.getI64IntegerAttr(1),
          ir.builder.getI64IntegerAttr(bytes), message);
      auto completion = ir.builder.create<mlir::async::AwaitOp>(
          ir.module->getLoc(), send.getToken());
      auto release = ir.builder.create<mlir::memref::DeallocOp>(
          ir.module->getLoc(), buffer);
      lifetimes.push_back({object,
                           wave,
                           definition.getOperation(),
                           {send.getOperation()},
                           {completion.getOperation()},
                           release.getOperation()});
    }
    EXPECT_TRUE(mlir::succeeded(
        verifySelectedStorageLifetimes(objects, lifetimes, &failureReason)))
        << failureReason;
    std::vector<SelectedStorageLifetimeBinding> early = lifetimes;
    early.front().release = early.front().uses.front();
    EXPECT_TRUE(mlir::failed(
        verifySelectedStorageLifetimes(objects, early, &failureReason)));
    ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
    EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
  }
}

TEST(StorageObjectBuilderTest,
     RealScaleSteadySelectorsFeedKAndActualMiniMalloc) {
  for (int64_t extent : {1024, 1025, 1031}) {
    for (uint32_t multiplicity : {2u, 3u, 4u, 5u}) {
      SCOPED_TRACE(extent);
      SCOPED_TRACE(multiplicity);
      IRFixture ir = makeIR();
      auto region = ir.builder.create<TileRegionOp>(
          ir.module->getLoc(), mlir::TypeRange{}, mlir::ValueRange{});
      region.getBody().push_back(new mlir::Block());
      mlir::OpBuilder storageBuilder =
          mlir::OpBuilder::atBlockEnd(&region.getBody().front());
      PhysicalVersionId version = makeVersion(0);
      StorageObjectId object{StorageObjectOrigin(version)};
      const uint64_t tail = extent % 128 == 0 ? 0 : 1;
      OccurrenceRelationId occurrence;
      occurrence.scope = TraversalScopeId{RegionExecutionId{makeExecution(0)},
                                          TopLevelWorkPieceId{0}};
      occurrence.axisOccurrences = {
          2, static_cast<uint64_t>((extent + 127) / 128), 1};
      PreparedStoragePlan prepared;
      prepared.objects.push_back(
          {{object, TileId(0)},
           StorageResourceDescription{object, makeDomain({2, extent, 128}),
                                      storageBuilder.getF16Type(),
                                      MemLayout::Tensor},
           multiplicity});
      prepared.bindings.push_back({version, object, StorageBindingKind::Fresh});
      prepared.slotFamilies.push_back(
          {SlotFamilyId{{object}}, occurrence, multiplicity, {1}});
      BufferPlan buffers;
      buffers.storageObjects.push_back({object, TileId(0)});
      buffers.versionBindings = prepared.bindings;
      buffers.slotFamilies = prepared.slotFamilies;

      StorageObjectBuilder objects;
      std::string failureReason;
      ASSERT_TRUE(mlir::succeeded(emitPreparedStorageObjects(
          prepared, objects, storageBuilder, &failureReason)))
          << failureReason;
      auto lower = storageBuilder.create<mlir::arith::ConstantIndexOp>(
          ir.module->getLoc(), 128);
      auto upper = storageBuilder.create<mlir::arith::ConstantIndexOp>(
          ir.module->getLoc(), 1024);
      auto step = storageBuilder.create<mlir::arith::ConstantIndexOp>(
          ir.module->getLoc(), 128);
      auto zero = storageBuilder.create<mlir::arith::ConstantIndexOp>(
          ir.module->getLoc(), 0);
      mlir::Value value = storageBuilder.create<mlir::arith::ConstantOp>(
          ir.module->getLoc(),
          storageBuilder.getFloatAttr(storageBuilder.getF16Type(), 0.0));
      auto loop = storageBuilder.create<mlir::scf::ForOp>(ir.module->getLoc(),
                                                          lower, upper, step);
      mlir::OpBuilder bodyBuilder =
          mlir::OpBuilder::atBlockBegin(loop.getBody());

      EventId storeEvent{ExecutionEventAction{makeExecution(10)},
                         PlannedEventKind::ComputeIssue};
      EventId loadEvent{ExecutionEventAction{makeExecution(11)},
                        PlannedEventKind::Completion};
      PipelineScopeId scope{{storeEvent, loadEvent}, {occurrence}};
      llvm::sort(scope.events);
      PipelinedExecutionStructure pipeline;
      pipeline.scope = scope;
      pipeline.recurrence = occurrence;
      pipeline.iteration = {1, 1, 7, tail};
      pipeline.eventStages = {{storeEvent, StageId(0)},
                              {loadEvent, StageId(1)}};
      llvm::sort(pipeline.eventStages);
      pipeline.dependences = {
          {storeEvent, loadEvent, 0, PipelineDependenceKind::Effect}};
      auto coordinate = buildSteadyOccurrenceCoordinate(
          pipeline, loop, bodyBuilder, &failureReason);
      ASSERT_TRUE(mlir::succeeded(coordinate)) << failureReason;
      auto selected = objects.select(object, /*recurrenceAxis=*/1, *coordinate,
                                     bodyBuilder, &failureReason);
      ASSERT_TRUE(mlir::succeeded(selected)) << failureReason;
      auto store = bodyBuilder.create<mlir::memref::StoreOp>(
          ir.module->getLoc(), value, selected->value,
          mlir::ValueRange{zero, zero, zero});
      auto load = bodyBuilder.create<mlir::memref::LoadOp>(
          ir.module->getLoc(), selected->value,
          mlir::ValueRange{zero, zero, zero});
      (void)load;
      std::vector<RotatingStorageSelection> selections{*selected};

      std::vector<mlir::Operation *> storeOperations;
      for (mlir::Operation &operation : loop.getBody()->without_terminator())
        if (&operation != load.getOperation())
          storeOperations.push_back(&operation);
      ExecutionStructureLoopBinding binding;
      binding.scope = scope;
      binding.steadyLoop = loop;
      binding.eventOrder = {{storeEvent, std::move(storeOperations)},
                            {loadEvent, {load}}};
      for (mlir::Value slot : objects.getSlots(object))
        binding.externalStorageProofs.push_back({slot, object});
      if (tail != 0) {
        auto tailCoordinate =
            storageBuilder.create<mlir::arith::ConstantIndexOp>(
                ir.module->getLoc(),
                static_cast<int64_t>(occurrence.axisOccurrences[1] - 1));
        auto tailSelection =
            objects.select(object, /*recurrenceAxis=*/1, tailCoordinate,
                           storageBuilder, &failureReason);
        ASSERT_TRUE(mlir::succeeded(tailSelection)) << failureReason;
        storageBuilder.create<mlir::memref::LoadOp>(
            ir.module->getLoc(), tailSelection->value,
            mlir::ValueRange{zero, zero, zero});
        selections.push_back(std::move(*tailSelection));
      }
      ASSERT_TRUE(mlir::succeeded(
          verifyRotatingStorageSelections(objects, selections, &failureReason)))
          << failureReason;
      if (extent == 1024 && multiplicity == 2) {
        RotatingStorageSelection broken = selections.front();
        broken.selectorOperations.pop_back();
        EXPECT_TRUE(mlir::failed(verifyRotatingStorageSelections(
            objects, {broken}, &failureReason)));
        unsigned selectorCount = 0;
        loop.walk([&](mlir::arith::SelectOp) { ++selectorCount; });
        EXPECT_TRUE(mlir::failed(objects.select(object, /*recurrenceAxis=*/0,
                                                *coordinate, bodyBuilder,
                                                &failureReason)));
        unsigned selectorCountAfter = 0;
        loop.walk([&](mlir::arith::SelectOp) { ++selectorCountAfter; });
        EXPECT_EQ(selectorCountAfter, selectorCount);
      }
      for (mlir::Value slot : objects.getSlots(object))
        storageBuilder.create<mlir::memref::DeallocOp>(ir.module->getLoc(),
                                                       slot);
      storageBuilder.create<TileYieldOp>(ir.module->getLoc());
      ir.builder.setInsertionPointAfter(region);
      ir.builder.create<mlir::func::ReturnOp>(ir.module->getLoc());
      ASSERT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));

      ExecutionStructurePlan structure{{pipeline}};
      PreparedExecutionStructureResult preparedStructure =
          prepareExecutionStructureMaterialization(*ir.module, structure,
                                                   buffers, {binding});
      ASSERT_TRUE(preparedStructure.succeeded())
          << (preparedStructure.failure ? preparedStructure.failure->detail
                                        : "");
      MaterializedExecutionStructureResult materialized =
          materializeExecutionStructure(std::move(ir.module),
                                        std::move(*preparedStructure.prepared));
      ASSERT_TRUE(materialized.succeeded())
          << (materialized.failure ? materialized.failure->detail : "");
      ir.module = std::move(materialized.materialized->module);
      EXPECT_TRUE(mlir::succeeded(wafer::planSPMMemoryModule(
          *ir.module, /*spmBase=*/0, /*spmLimit=*/3 * 1024 * 1024,
          /*spmAlignment=*/16)));
      unsigned allocations = 0;
      ir.module->walk([&](mlir::memref::AllocOp) { ++allocations; });
      EXPECT_EQ(allocations, multiplicity);
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*ir.module)));
    }
  }
}

} // namespace
